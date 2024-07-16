import string
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
import numpy as np

from utils import get_args, CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

# Static files configuration to serve CSS, JS, etc.
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and configurations
opt = get_args()
import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

class DTRB:
    def __init__(self):
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        """ model configuration """
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(opt.saved_model, opt)

    def load_model(self, weights_path, opt):
        self.model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % weights_path)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def predict(self, image, opt):
        transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
        # predict
        self.model.eval()
        with torch.no_grad():
            image_tensor = transform(image)
            image_tensor = image_tensor.sub_(0.5).div_(0.5)
            image_tensor = torch.unsqueeze(image_tensor, 0)  # output: [1, 1, 32, 100]
            
            # for image_tensors, image_path_list in self.demo_loader:
            batch_size = image_tensor.size(0)
            image = image_tensor.to(self.device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(self.device)

            if 'CTC' in opt.Prediction:
                preds = self.model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index, preds_size)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            # log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(["besco"], preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                # log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            # log.close()


plate_recognizer = DTRB()

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <title>OCR Image Upload</title>
        </head>
        <body>
            <h1>Upload an image for OCR</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit">
            </form>
        </body>
    </html>
    """
    return content

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plate_recognizer.predict(image, opt)

            # Process result
        result = []
            # for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            #     if 'Attn' in opt.Prediction:
            #         pred_EOS = pred.find('[s]')
            #         pred = pred[:pred_EOS]
            #         pred_max_prob = pred_max_prob[:pred_EOS]
            #     confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
            #     result.append({"text": pred, "confidence": confidence_score})

        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
