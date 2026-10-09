import re
import cv2
import time
import rospy
import rospkg
import numpy as np
import skimage as ski
import polylabel

from skimage.segmentation import slic
from sensor_msgs.msg import Image, CompressedImage

# ChatGPT imports
import base64
from openai import OpenAI

class LLMPredictor:
    def __init__(self):
        # Parameters
        input_topic = rospy.get_param("~input_topic", "/terrasentia/zed2/zed_node/left/image_rect_color/compressed")
        output_topic = rospy.get_param("~output_topic", "/traversability_map/compressed")
        self.image_resolution = rospy.get_param('~image_resolution', [336, 224])
        self.area_threshold = rospy.get_param('~area_threshold', 500)
        self.test_mode = rospy.get_param('~test_mode', True)

        # Load the ChatGPT model
        key_file = rospkg.RosPack().get_path('llm_prediction') + '/../api_key.txt'
        with open(key_file, 'r') as file:
             api_key = file.read().strip()
        self.client = OpenAI(api_key=api_key)

        # Subscriber
        self.image_sub = rospy.Subscriber(input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)

        # Publisher
        self.image_pub = rospy.Publisher(output_topic, Image, queue_size=1)
        self.debug_pub = rospy.Publisher('numbered_image', Image, queue_size=1)

        rospy.spin()

    def query_gpt(self, image):
        """
        Queries the GPT model with the image.

        Parameters:
        - image: Image from the camera.

        Returns:
        - Ranks of the masks.
        """
        # Encode the image
        # with open(image_path, "rb") as image_file:
        #     base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        base64_image = base64.b64encode(cv2.imencode('.png', image)[1]).decode('utf-8')

        # Query the model
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            # model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": "In the image, analyze the numbered regions and assign traversability values for each region. \
                        The traversability values should be real numbers where 0 represents low traversability and 1 represents high traversability. \
                        For example, traversability 0.0 represents obstacles and dangerous areas like walls, trees, cars, and etc. \
                        Traversability 1.0 represents highly traversable areas for a small wheeled robot, like roads and walkways. \
                        The output should be the traversavbility values from region 1 to n without the reasoning, formatted as a list, \
                        for example [0,0.5,1,0.3,...]"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/png;base64,{base64_image}"
                    },
                    },
                ],
                }
            ],
            max_tokens=300
        )

        ranks = response.choices[0].message.content

        return ranks
    
    def add_numbers_to_masks(self, anns, image):
        """
        Adds numbers to corresponding masks on the image.
        
        Parameters:
        - anns: List of SLIC region masks and their areas.
        
        Returns:
        - Annotated image with numbers on the masks.
        """
        # If there are no annotations, return
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)[:10]
        
        rank_dict = {}
        for i, ann in enumerate(sorted_anns):
            m = ann['segmentation']

            # Apply the mask to the image
            labeled_image = ski.measure.label(m, connectivity=2, return_num=False)
            object_features = ski.measure.regionprops(labeled_image)
            largest_region = max(object_features, key=lambda x: x.area)

            # Create a binary mask of the largest region
            largest_mask = (labeled_image == largest_region.label).astype(np.uint8)

            # Find contours in the mask
            contours, _ = cv2.findContours(largest_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                rank_dict[i] = m
            
                # # Step 1: Create an empty mask of same size as 'm'
                # simplified_mask = np.zeros_like(m, dtype=np.uint8)

                # # Step 2: Draw the simplified contour on this empty mask
                # cv2.drawContours(simplified_mask, [approx_contour], -1, color=1, thickness=cv2.FILLED)

                # # Step 3: Store this simplified boolean mask in rank_dict
                # rank_dict[i] = simplified_mask.astype(bool)
            
                filtered = [list(map(lambda x: x[0], approx_contour))]
                pole_of_inaccessibility = polylabel.polylabel(filtered, precision=10.0)
                cX, cY = pole_of_inaccessibility

                text = str(i + 1)
                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                fontColor = (255, 0, 0)
                thickness = 2

                text_width, text_height = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
                cv2.putText(image, text, (int(cX - text_width/2), int(cY + text_height/2)), fontFace, 1, fontColor, thickness, cv2.LINE_AA)

        return rank_dict, image

    def image_callback(self, image_msg):
        """
        Callback function for the image subscriber.

        Parameters:
        - image: Compressed image from the camera.
        """
        # Convert the compressed image to a numpy array
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Get the original image resolution
        original_resolution = cv_image.shape[:2]

        # Resize the image
        cv_image = cv2.resize(cv_image, self.image_resolution)

        time_start = time.time()
        masks = slic(cv_image, n_segments=10, compactness=5, sigma=1)
        masks = [{'segmentation': masks == i, 'area': np.sum(masks == i)} for i in np.unique(masks)]
        print(f"Time taken for SLIC: {time.time() - time_start}")

        time_start = time.time()
        # Add numbers to the masks
        rank_dict, image = self.add_numbers_to_masks(masks, cv_image)
        print(f"Time taken to add the numbers: {time.time() - time_start}")

        # Query the GPT model
        time_start = time.time()
        ranks = self.query_gpt(image)
        print('Ranks:', ranks)
        print(f"Time taken for GPT: {time.time() - time_start}")

        # Resize the image to the original resolution
        image = cv2.resize(image, (original_resolution[1], original_resolution[0]))

        # Convert the image to a ROS message
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        debug_msg = Image()
        debug_msg.header = image_msg.header
        debug_msg.encoding = "bgr8"
        debug_msg.height = image.shape[0]
        debug_msg.width = image.shape[1]
        debug_msg.step = image.shape[1] * 3
        debug_msg.data = image_bgr.tobytes()

        # Publish the annotated image
        self.debug_pub.publish(debug_msg)

        # Parse the ranks
        # ranked_list = list(map(float, ranks.split(',')))
        ranked_list = re.split(';|,|\*|\n', ranks.replace(' ', '').replace('[', '').replace(']', ''))
        print('Ranked list:', ranked_list)

        # Create a traversability image
        # Note: the zero value is reserved for non-annotated regions
        trav_img = np.ones((self.image_resolution[1], self.image_resolution[0]), dtype=np.float32) * np.nan

        for i, rank in enumerate(ranked_list):
            if i in rank_dict:
                try:
                    rank = float(rank)
                except ValueError:
                    rospy.logwarn(f"Invalid rank value: {rank}")
                    rospy.logwarn("Skipping the image")
                    return
                
                trav_img[rank_dict[i]] = rank

        # Resize the traversability image to the original resolution
        trav_img = cv2.resize(trav_img, (original_resolution[1], original_resolution[0]), interpolation=cv2.INTER_NEAREST)

        # Convert the traversability image ros msg
        output_msg = Image()
        output_msg.header = image_msg.header
        output_msg.encoding = "32FC1"
        output_msg.height = trav_img.shape[0]
        output_msg.width = trav_img.shape[1]
        output_msg.step = trav_img.shape[1] * 4
        # We multiply by 100 just to visualize the traversability values better in rviz
        output_msg.data = (trav_img * 100).tobytes()
        
        # Publish the traversability image
        self.image_pub.publish(output_msg)
    
if __name__ == '__main__':
    rospy.init_node('llm_prediction')
    LLMPredictor()
