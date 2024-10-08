import cv2
import time
import torch
import rospy
import rospkg
import numpy as np
import skimage as ski
import polylabel

from torch.cuda.amp import autocast

from sensor_msgs.msg import Image, CompressedImage

# Import the SAM model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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
        model_type = rospy.get_param('~model_type', 'vit_b')

        # Load the SAM model
        if model_type == "vit_b":
            sam_checkpoint = rospkg.RosPack().get_path('llm_prediction') + "/checkpoint/sam_vit_b_01ec64.pth"
        elif model_type == "vit_l":
            sam_checkpoint = rospkg.RosPack().get_path('llm_prediction') + "/checkpoint/sam_vit_l_0b3195.pth"
        elif model_type == "vit_h":
            sam_checkpoint = rospkg.RosPack().get_path('llm_prediction') + "/checkpoint/sam_vit_h_4b8939.pth"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"LLM prediction node using device: {device}")

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)


        self.mask_generator = SamAutomaticMaskGenerator(
            model                   = sam,
            box_nms_thresh          = 0.7,
            crop_nms_thresh         = 0.7,
            crop_overlap_ratio      = 0.35,
            points_per_side         = 8,
            pred_iou_thresh         = 0.95,         # High confidence score
            stability_score_thresh  = 0.9,          # Compensate a bit on stability to capture big masks
            crop_n_layers           = 1,
            crop_n_points_downscale_factor = 4,
            min_mask_region_area    = 500,          # Requires open-cv to run post-processing
        )

        # Load the ChatGPT model
        key_file = rospkg.RosPack().get_path('llm_prediction') + '/../api_key.txt'
        with open(key_file, 'r') as file:
             api_key = file.read().strip()
        self.client = OpenAI(api_key=api_key)

        # Subscriber
        self.image_sub = rospy.Subscriber(input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)

        # Publisher
        self.image_pub = rospy.Publisher(output_topic, CompressedImage, queue_size=1)
        self.debug_pub = rospy.Publisher('numbered_image/compressed', CompressedImage, queue_size=1)

        rospy.spin()

    def query_gpt(self, image_path):
        """
        Queries the GPT model with the image.

        Parameters:
        - image: Image from the camera.

        Returns:
        - Ranks of the masks.
        """
        # Encode the image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
       
        # Query the model
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": "In the image, analyze the numbered regions and assign traversability values for each region. \
                        The traversability values should be real numbers where 0 represents low traversability and 1 represents high traversability. \
                        For example, traversability 0.0 represents obstacles and dangerous areas like trees, cars, and etc. \
                        Traversability 1.0 represents highly traversable areas like roads and flat terrains. \
                        Assume the image comes from a small wheeled robot. \
                        The output should be the traversavbility values from region 1 to n without the reasoning, for example 0,0.5,1,..."},
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
        - anns: List of annotations from the model.
        
        Returns:
        - Annotated image with numbers on the masks.
        """
        # If there are no annotations, return
        if len(anns) == 0:
            return

        anns = [ann for ann in anns if ann['area'] > 500]
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)[:10]
        
        img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1],4))
        new_masks = []

        while sorted_anns:
            bitwise_or = [cv2.bitwise_and(ann['segmentation'].astype(int),sorted_anns[0]['segmentation'].astype(int))for ann in sorted_anns]
            any = np.array([np.any(a) for a in bitwise_or])
            idx = np.where(any)[0]
            areas = [sorted_anns[i]['area'] for i in idx]
            largest = np.argmax(areas)
            new_masks.append(sorted_anns[largest])
            sorted_idx = np.sort(idx)[::-1]
            for i in sorted_idx:
                del sorted_anns[i]
#
        rank_dict = {}
        for i, ann in enumerate(new_masks):
            m = ann['segmentation']

            # Apply the mask to the image
            labeled_image = ski.measure.label(m, connectivity=2, return_num=False)
            object_features = ski.measure.regionprops(labeled_image)
            largest_region = max(object_features, key=lambda x: x.area)
            largest_mask = np.zeros_like(m)
            largest_mask[labeled_image == largest_region ["label"]] = 1

            # Convert mask to uint8 type for cv2.findContours
            mask_uint8 = np.uint8(largest_mask* 255)  # Scale mask to 0-255 for OpenCV
            # Find contours in the mask
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
        
            filtered = [list(map(lambda x: x[0], largest_contour))]
            pole_of_inaccessibility = polylabel.polylabel(filtered, precision=1.0)
            cX, cY = pole_of_inaccessibility

            text = str(i + 1)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (255, 0, 0)
            thickness = 2

            text_width, text_height = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
            cv2.putText(image, text, (int(cX - text_width/2), int(cY + text_height/2)), fontFace, 1, fontColor, thickness, cv2.LINE_AA)
            rank_dict[i] = m

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
        # Get masks from the SAM model
        with autocast():
            masks = self.mask_generator.generate(cv_image)
        print(f"Time taken for SAM: {time.time() - time_start}")

        time_start = time.time()
        # Add numbers to the masks
        rank_dict, image = self.add_numbers_to_masks(masks, cv_image)
        cv2.imwrite("img.png",image)
        print(f"Time taken to add the numbers: {time.time() - time_start}")

        # Resize the image to the original resolution
        image = cv2.resize(image, (original_resolution[1], original_resolution[0]))

        # Convert the image to a ROS message
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, im_encoded = cv2.imencode('.jpg', image_bgr)

        debug_msg = CompressedImage()
        debug_msg.header = image_msg.header
        debug_msg.format = "jpeg"
        debug_msg.data = np.array(im_encoded).tobytes()

        # Publish the annotated image
        self.debug_pub.publish(debug_msg)

        time_start = time.time()
        # Query the GPT model 
        ranks = self.query_gpt("img.png")
        print('Ranks:', ranks)
        print(f"Time taken for GPT: {time.time() - time_start}")

        # Parse the ranks
        ranked_list = list(map(float, ranks.split(',')))

        # Create a traversability image
        # Note: the zero value is reserved for non-annotated regions
        trav_img = np.zeros((self.image_resolution[1], self.image_resolution[0]))

        for i, rank in enumerate(ranked_list):
            if i in rank_dict:
                trav_img[rank_dict[i]] = rank

        # Resize the traversability image to the original resolution
        trav_img = cv2.resize(trav_img, (original_resolution[1], original_resolution[0]))

        # Convert the traversability image ros msg
        trav_img = np.uint8(trav_img * 255)

        _, im_encoded = cv2.imencode('.jpg', trav_img)
        output_msg = CompressedImage()
        output_msg.header = image_msg.header
        output_msg.format = "jpeg"
        output_msg.data = np.array(im_encoded).tobytes()
        
        # Publish the traversability image
        self.image_pub.publish(output_msg)
    
if __name__ == '__main__':
    rospy.init_node('llm_prediction')
    LLMPredictor()
