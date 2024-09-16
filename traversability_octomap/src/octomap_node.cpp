#include "octomap_node.h"

static const float MIN_TRAVERSABILITY = 0.0;
static const float MAX_TRAVERSABILITY = 1.0;

OctomapNode::OctomapNode()
    : nh_("~"),
      tfListener_(tfBuffer_)
{
    // Initialize parameters
    initParameters();

    // Initialize map and publisher objects
    map_ = std::make_unique<traversability_bki::TraversabilityBKIOctoMap>(
        resolution_, block_depth_, sf2_, ell_, mean_prior_, lambda_prior_,
        beta_prior_, prior_A_, prior_B_, var_thresh_, free_thresh_,
        occupied_thresh_, min_z_, max_z_);

    marker_pub_ = std::make_unique<traversability_bki::MarkerArrayPub>(
        nh_, map_topic_, resolution_, block_depth_, frame_id_);

    // Set up publishers and subscribers
    grid_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>(grid_map_topic_, 1);
    point_cloud_sub_ = nh_.subscribe(pointcloud_topic_, 5, &OctomapNode::pointCloudCallback, this);

    // Precompute the Student's t-distribution quantiles
    int num_quantiles = 100;
    std::vector<double> precomputed_quantiles;
    for (double i = 1; i <= nu_threshold_; i += nu_threshold_ / num_quantiles) {
        boost::math::students_t dist(i);
        double t_quantile = boost::math::quantile(dist, confidence_level_);
        precomputed_quantiles.push_back(t_quantile);
    }
    quantile_interp_ = boost::math::cubic_b_spline<double>(precomputed_quantiles.begin(), precomputed_quantiles.end(), 1.0, 1.0);

    // Precompute the Student's t-distribution PDF values
    std::vector<double> precomputed_pdf;
    for (double i = 1; i <= nu_threshold_; i += nu_threshold_ / num_quantiles) {
        boost::math::students_t dist(i);
        double t_quantile = boost::math::quantile(dist, confidence_level_);
        precomputed_pdf.push_back(boost::math::pdf(dist, t_quantile));
    }
    pdf_interp_ = boost::math::cubic_b_spline<double>(precomputed_pdf.begin(), precomputed_pdf.end(), 1.0, 1.0);

    ROS_INFO("OctomapNode initialized, waiting for point cloud data...");
}

void OctomapNode::spin()
{
    ros::spin();
}

void OctomapNode::initParameters()
{
    nh_.param<int>("block_depth", block_depth_, block_depth_);
    nh_.param<double>("sf2", sf2_, sf2_);
    nh_.param<double>("ell", ell_, ell_);
    nh_.param<float>("mean_prior", mean_prior_, mean_prior_);
    nh_.param<float>("lambda_prior", lambda_prior_, lambda_prior_);
    nh_.param<float>("beta_prior", beta_prior_, beta_prior_);
    nh_.param<float>("prior_A", prior_A_, prior_A_);
    nh_.param<float>("prior_B", prior_B_, prior_B_);
    nh_.param<float>("var_thresh", var_thresh_, var_thresh_);
    nh_.param<double>("free_thresh", free_thresh_, free_thresh_);
    nh_.param<double>("occupied_thresh", occupied_thresh_, occupied_thresh_);
    nh_.param<float>("resolution", resolution_, resolution_);
    nh_.param<double>("ds_resolution", ds_resolution_, ds_resolution_);
    nh_.param<double>("free_resolution", free_resolution_, free_resolution_);
    nh_.param<double>("min_range", min_range_, min_range_);
    nh_.param<double>("max_range", max_range_, max_range_);
    nh_.param<float>("min_z", min_z_, min_z_);
    nh_.param<float>("max_z", max_z_, max_z_);
    nh_.param<float>("min_z_grid", min_z_grid_, min_z_grid_);
    nh_.param<float>("max_z_grid", max_z_grid_, max_z_grid_);
    nh_.param<float>("confidence_level", confidence_level_, confidence_level_);
    nh_.param<std::string>("frame_id", frame_id_, std::string("odom"));
    nh_.param<std::string>("pointcloud_topic", pointcloud_topic_, std::string("traversability_cloud"));
    nh_.param<std::string>("map_topic", map_topic_, std::string("traversability_cells"));
    nh_.param<std::string>("grid_map_topic", grid_map_topic_, std::string("traversability_map"));
}

void OctomapNode::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
{
    geometry_msgs::TransformStamped transformStamped;

    try {
        transformStamped = tfBuffer_.lookupTransform(frame_id_, cloud->header.frame_id, cloud->header.stamp, ros::Duration(10));
    }
    catch (tf2::ConnectivityException &ex) {
        ROS_ERROR("ConnectivityException: %s", ex.what());
        return;
    }
    catch (tf2::TransformException &ex) {
        ROS_ERROR("Transform failed: %s", ex.what());
        return;
    }

    ros::Time start = ros::Time::now();
    traversability_bki::point3f origin;
    tf2::Vector3 translation = tf2::Vector3(
        transformStamped.transform.translation.x,
        transformStamped.transform.translation.y,
        transformStamped.transform.translation.z
    );
    tf2::Quaternion orientation = tf2::Quaternion(
        transformStamped.transform.rotation.x,
        transformStamped.transform.rotation.y,
        transformStamped.transform.rotation.z,
        transformStamped.transform.rotation.w
    );

    if (first_ || orientation.angleShortestPath(last_orientation_) > orientation_change_thresh_ || translation.distance(last_position_) > position_change_thresh_) 
    {
        ROS_INFO_STREAM("Cloud received");
        
        last_position_ = translation;
        last_orientation_ = orientation;
        origin = traversability_bki::point3f(translation.x(), translation.y(), translation.z());

        sensor_msgs::PointCloud2 cloud_map;
        pcl_ros::transformPointCloud(frame_id_, *cloud, cloud_map, tfBuffer_);

        //pointer required for downsampling
        traversability_bki::PCLPointCloud pcl_cloud;
        pcl::fromROSMsg(cloud_map, pcl_cloud);

        if(pcl_cloud.size() > 5)
        {
            map_->insert_pointcloud(pcl_cloud, origin, (float) ds_resolution_, (float) free_resolution_, (float) min_range_, (float) max_range_);
        }

        ROS_INFO_STREAM("One cloud finished in " << (ros::Time::now() - start).toSec() << "s");
        updated_ = true;
    }

    if (updated_) 
    {
        ROS_INFO_STREAM("Cloud received for update");
        start = ros::Time::now();

        int num_voxels = 0;
        marker_pub_->clear();
        for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it)
        {
            traversability_bki::point3f p = it.get_loc();
            if (it.get_node().get_state() == traversability_bki::State::OCCUPIED)
            {
                double traversability = it.get_node().get_mean();
                if (traversability < MIN_TRAVERSABILITY || traversability > MAX_TRAVERSABILITY)
                {
                    ROS_WARN_STREAM("Traversability out of bounds: " << traversability);
                }
                marker_pub_->insert_color_point3d(p.x(), p.y(), p.z(), MIN_TRAVERSABILITY, MAX_TRAVERSABILITY, it.get_size(), traversability);
                num_voxels++;
            }
        }
        updated_ = false;

        ROS_INFO_STREAM("Publishing map with " << num_voxels << " voxels");
        marker_pub_->publish();
        ROS_INFO_STREAM("Octomap published in " << (ros::Time::now() - start).toSec() << "s");

        start = ros::Time::now();
        publishGridMap(); // Publish the grid map
        ROS_INFO_STREAM("Grid map published in " << (ros::Time::now() - start).toSec() << "s");
    }
}

void OctomapNode::publishGridMap()
{
    // Create the grid map with one layer named "traversability"
    grid_map::GridMap map({"traversability", "lambda"});
    map.setFrameId(frame_id_);
    traversability_bki::point3f lim_min, lim_max;
    map_->get_bbox(lim_min, lim_max);
    grid_map::Length map_size(lim_max.x()-lim_min.x(), lim_max.y()-lim_min.y());
    grid_map::Position map_position((lim_max.x()+lim_min.x())/2, (lim_max.y()+lim_min.y())/2);
    map.setGeometry(map_size, resolution_, map_position);

    // Initialize the grid map with default values
    float sigma_prior = sqrt(2 * beta_prior_ / lambda_prior_);
    float cvar_default = calculateCVaR(mean_prior_, sigma_prior, lambda_prior_, confidence_level_);
    map.add("traversability", cvar_default);
    map.add("lambda", lambda_prior_);

    // A map to store traversability values for each cell for CVaR computation
    std::unordered_map<grid_map::Index, std::vector<float>, std::hash<grid_map::Index>, std::equal_to<grid_map::Index>> NIG_values;

    ros::Time start = ros::Time::now();
    // Iterate through the octomap and populate the grid map
    for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it)
    {
        if (it.get_node().get_state() == traversability_bki::State::OCCUPIED)
        {
            traversability_bki::point3f p = it.get_loc();
            grid_map::Position position(static_cast<double>(p.x()), static_cast<double>(p.y()));
            if (map.isInside(position) && p.z() >= min_z_grid_ && p.z() <= max_z_grid_)
            {
                grid_map::Index index;
                map.getIndex(position, index);

                // Get the mean of the node
                float mean = it.get_node().get_mean();

                // Get the lambda of the node
                float lambda = it.get_node().get_lambda();

                // Get the beta of the node
                float beta = it.get_node().get_beta();

                // Calculate the CVaR for the cell
                float sigma = sqrt(2 * beta / lambda);

                float cvar = calculateCVaR(mean, sigma, lambda, confidence_level_);

                // Check if the cell is already in the map
                if (NIG_values.find(index) != NIG_values.end())
                {
                    // Get the current mean
                    float curr_cvar = NIG_values[index][0];

                    // Get the minimum CVaR value
                    if (cvar < curr_cvar) {
                        NIG_values[index] = {cvar, lambda};
                    }                    
                }
                // If the cell is not in the map
                else
                {
                    NIG_values[index] = {cvar, lambda};
                }
            }
        }
    }
    ROS_INFO_STREAM("Grid map CVaR loop takes " << (ros::Time::now() - start).toSec() << "s");

    // Now pass the hashmap values to the grid map
    for (const auto& item : NIG_values)
    {
        const auto& index = item.first;
        const auto& values = item.second;
        double cvar = values[0];
        double lambda = values[1];

        // Update the grid map with the CVaR value
        map.at("traversability", index) = cvar;
        map.at("lambda", index) = lambda;
    }

    // Publish the grid map
    grid_map_msgs::GridMap message;
    grid_map::GridMapRosConverter::toMessage(map, message);
    grid_map_pub_.publish(message);
}

double OctomapNode::calculateCVaR(double mu, double sigma, double nu, double alpha) {
    double cvar;
    // If nu is low, calculate CVaR using the Student's t-distribution
    if (nu < nu_threshold_) {
        // Create Student's t-distribution object
        boost::math::students_t dist(nu);
        
        // Compute t_quantile
        // double t_quantile = boost::math::quantile(dist, alpha);
        double t_quantile = quantile_interp_(nu);
        
        // Evaluate PDF at t_quantile
        double pdf_at_t = pdf_interp_(nu);
        
        // Compute CVaR
        cvar = mu - sigma * (nu + t_quantile * t_quantile) / (nu - 1.0) * pdf_at_t / (alpha);
    }
    // If nu is high, approximate CVaR using normal distribution
    else {
        // Create Gaussian distribution object
        boost::math::normal dist;

        // Compute quantile
        double g_quantile = boost::math::quantile(dist, alpha);

        // Evaluate PDF at quantile
        double pdf_at_t = boost::math::pdf(dist, g_quantile);
        
        cvar = mu - sigma * pdf_at_t / alpha;
    }

    // std::cout << "mu: " << mu << ", CVaR: " << cvar  << ", t_quantile: " << t_quantile << ", pdf_at_t: " << pdf_at_t << ", nu: " << nu << std::endl;
    
    return cvar;
}