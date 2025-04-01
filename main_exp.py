import cv2
import torch
from torch.nn import Softmax
from torchvision import transforms
from EXP_module.src.model import NLA_r18
from EXP_module.src.utils import *
from EXP_module.src.resnet import *
from face_det_module.src.util import get_args_parser, get_transform
from face_det_module.src.face_crop import crop

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
v_font_color = (0, 255, 0)  
a_font_color = (0, 0, 255)  
font_thickness = 2
resize_param = 250
exp_dict = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Neutral'}
colors = ['orange', 'purple', 'green', 'yellow', 'blue', 'red', 'gray']

if __name__ == "__main__":
    args = get_args_parser()
    args.transform = get_transform()
    args.weights_path = '/NLA/Expression_Classification/EXP_module/weights/best.pth'
    model = NLA_r18(args)
    cp = torch.load(args.weights_path)
    model.load_state_dict(cp)
    model = model.to('cuda')
    model.eval()

    idx = True
    cap = cv2.VideoCapture(0)
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                         std=[0.2628, 0.2395, 0.2383])])
    while idx:
        idx, image = cap.read()
        image = cv2.flip(image, 1)
        img_height, img_width = image.shape[:2]
        output_image, check = crop(image, preprocess, 224, True, 'cuda')
        if check:
            output = model(output_image)
            softmax = Softmax(dim=1)
            probabilities = softmax(output)
            probabilities = probabilities.squeeze(0).detach().cpu().numpy()
            predicted_class = np.argmax(probabilities)
            predicted_label = exp_dict[predicted_class]

            cv2.putText(image, "Exp: " + str(predicted_label), (10, 50), font, font_size, v_font_color, font_thickness)
        else:

             cv2.putText(image, "Exp: " + 'None', (10, 50), font, font_size, a_font_color, font_thickness)

        


        
        if idx == False:
            cap.release()
        else:
            cv2.imshow("Output", image)
            k = cv2.waitKey(2) & 0xFF
            if k == 27: # ESC key
                break