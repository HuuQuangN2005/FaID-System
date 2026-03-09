import torch
import cv2
import os
from torchvision import transforms
from model import LandmarkModel

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

class LandmarkInference:

    def __init__(self, weight_path):

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"WF not found: {weight_path}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = LandmarkModel().to(self.device)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.result_dir = os.path.join(
            PROJECT_ROOT,
            "models",
            "alignment",
            "results"
        )

        os.makedirs(self.result_dir, exist_ok=True)


    def predict(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():

            output = self.model(input_tensor)

        landmarks = output.cpu().numpy().reshape(-1, 2)

        landmarks[:, 0] *= w
        landmarks[:, 1] *= h

        return img, landmarks

    def save_result(self, image_path):

        img, landmarks = self.predict(image_path)

        for (x, y) in landmarks:
            cv2.circle(img, (int(x), int(y)), 2, (0, 180, 0), -1)

        eye_left = landmarks[0]
        eye_right = landmarks[1]

        cv2.line(
            img,
            (int(eye_left[0]), int(eye_left[1])),
            (int(eye_right[0]), int(eye_right[1])),
            (0, 0, 255),
            1
        )

        filename = os.path.basename(image_path)
        save_path = os.path.join(self.result_dir, filename)
        cv2.imwrite(save_path, img)

        print("Saved:", save_path)


if __name__ == "__main__":

    weight_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "alignment",
        "weights",
        "best_model.pth"
    )
    #
    # image_path = os.path.join(
    #     PROJECT_ROOT,
    #     "data",
    #     "alignment",
    #     "raw",
    #     "img_align_celeba",
    #     "052102.jpg"
    # )

    image_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "alignment",
        "test",
        "02776.jpg"
    )

    infer = LandmarkInference(weight_path)
    infer.save_result(image_path)