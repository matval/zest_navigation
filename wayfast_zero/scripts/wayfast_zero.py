import cv2
import time
import rospy
import rospkg
import numpy as np

from torch.cuda.amp import autocast

from sensor_msgs.msg import Image, CompressedImage

# Import the SAM model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ChatGPT imports
import base64
from openai import OpenAI

class WayFastZero:
    def __init__(self):
        # Parameters
        input_topic = rospy.get_param("input_topic", "/camera/image/compressed")
        output_topic = rospy.get_param("output_topic", "/traversability_map")
        self.image_resolution = rospy.get_param('image_resolution', [336, 224])
        self.area_threshold = rospy.get_param('area_threshold', 500)
        self.test_mode = rospy.get_param('test_mode', True)
        model_type = rospy.get_param('model_type', 'vit_b')

        # Load the SAM model
        if model_type == "vit_b":
            sam_checkpoint = rospkg.RosPack().get_path('wayfast_zero') + "/checkpoint/sam_vit_b_01ec64.pth"
        elif model_type == "vit_l":
            sam_checkpoint = rospkg.RosPack().get_path('wayfast_zero') + "/checkpoint/sam_vit_l_0b3195.pth"
        elif model_type == "vit_h":
            sam_checkpoint = rospkg.RosPack().get_path('wayfast_zero') + "/checkpoint/sam_vit_h_4b8939.pth"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model                   = sam,
            box_nms_thresh          = 1,
            crop_nms_thresh         = 0.7,
            crop_overlap_ratio      = 0.35,
            points_per_side         = 16,
            pred_iou_thresh         = 0.95,         # High confidence score
            stability_score_thresh  = 0.9,          # Compensate a bit on stability to capture big masks
            crop_n_layers           = 1,
            crop_n_points_downscale_factor = 2,
            min_mask_region_area    = 500,          # Requires open-cv to run post-processing
        )

        # Load the ChatGPT model
        if not self.test_mode:
            key_file = rospkg.RosPack().get_path('wayfast_zero') + '../api_key.txt'
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
        base64_image = base64.b64encode(image).decode('utf-8')

        if self.test_mode:
            return "1,2,3,4,5"
        
        # Query the model
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "rank the numbers in the image in terms of traversability for robots and just give the numbers"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":f"data:image/jpeg;base64,{base64_image}"
                    },
                    },
                ],
                }
            ],
            max_tokens=100
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
        
        # Filter the annotations based on area
        filtered_anns = [ann for ann in anns if ann['area'] > self.area_threshold]
        sorted_anns = sorted(filtered_anns, key=(lambda x: x['area']), reverse=True)[:10]

        rank_dict = {}

        for i, ann in enumerate(sorted_anns, start=1):
            m = ann['segmentation']
            y, x = np.where(m)

            if len(y) > 0 and len(x) > 0:
                # Compute the center of the mask
                cY, cX = int(np.mean(y)), int(np.mean(x))

                # Add the number at the center of the mask
                cv2.putText(image, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Return a dictionary with key:mask number and value:corresponding mask
                rank_dict[str(i)] = m

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

        # Resize the image
        cv_image = cv2.resize(cv_image, self.image_resolution)

        time_start = time.time()
        # Get masks from the SAM model
        with autocast():
            masks = self.mask_generator.generate(cv_image)
        print(f"Time taken for SAM: {time.time() - time_start}")

        # Add numbers to the masks
        rank_dict, image = self.add_numbers_to_masks(masks, cv_image)

        # Convert the image to ros msg
        image_msg = Image()
        image_msg.header = image_msg.header
        image_msg.encoding = 'rgb8'
        image_msg.height = image.shape[0]
        image_msg.width = image.shape[1]
        image_msg.step = image.shape[1] * 3
        image_msg.data = image.astype('uint8').flatten().tolist()

        # Publish the annotated image
        self.debug_pub.publish(image_msg)

        time_start = time.time()
        # Query the GPT model
        ranks = self.query_gpt(cv_image)
        print(f"Time taken for GPT: {time.time() - time_start}")

        # Parse the ranks
        ranked_list = list(map(int, ranks.split(',')))

        ranked_dict = {}
        for i, rank in enumerate(ranked_list):
            ranked_dict[rank]= i+1

        # Create a traversability image
        # Note: the zero value is reserved for non-annotated regions
        trav_img = np.zeros(image.shape[:2])

        max_rank = max(ranked_dict.values())
        for i, key in ranked_dict.items():
            if str(i) in rank_dict:
                trav_img[rank_dict[str(i)]] = key / max_rank

        # Convert the traversability image ros msg
        trav_img = np.uint8(trav_img * 255)

        output_msg = Image()
        output_msg.header = image_msg.header
        output_msg.encoding = 'mono8'
        output_msg.height = trav_img.shape[0]
        output_msg.width = trav_img.shape[1]
        output_msg.step = trav_img.shape[1]
        output_msg.data = trav_img.flatten().tolist()
        
        # Publish the traversability image
        self.image_pub.publish(output_msg)
    
if __name__ == '__main__':
    rospy.init_node('wayfast_zero')
    WayFastZero()