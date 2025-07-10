import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import EPOCHS, ALPHA
from dataset import get_dataloaders
from model import get_model
from feedback import init_cam, selective_indices, generate_cams, ask_llm

def train():
    train_dl, test_dl = get_dataloaders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cam_extractor = init_cam(model)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # counters for this epoch
        feedback_batches = 0
        feedback_images  = 0
        running_loss     = 0.0

        # wrap training loader with tqdm
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.squeeze().long().to(device)

            out   = model(xb)
            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(dim=1)

            loss = F.cross_entropy(out, yb)

            select_idx = selective_indices(probs)
            if len(select_idx) > 0:
                feedback_batches += 1
                feedback_images  += len(select_idx)

                selected_imgs  = xb[select_idx]
                selected_preds = preds[select_idx]
                selected_confs = probs[select_idx].max(dim=1).values.tolist()

                cam_images = generate_cams(cam_extractor, model, selected_imgs, selected_preds)
                scores     = ask_llm(cam_images, selected_preds.tolist(), selected_confs)

                feedback_loss = (1 - scores).mean()
                loss         += ALPHA * feedback_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (pbar.n + 1)

            pbar.set_postfix({
                "loss":    f"{avg_loss:.4f}",
                "fb_bs":   feedback_batches,
                "fb_imgs": feedback_images
            })

        pbar.close()

        # epoch summary
        print(
            f"\nEpoch {epoch}/{EPOCHS} complete — "
            f"Avg Loss: {running_loss/len(train_dl):.4f}  "
            f"LLM feedback on {feedback_batches} batches "
            f"({feedback_images} images)\n"
        )
        evaluate(model, test_dl, device)

def evaluate(model, dl, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.squeeze().long().to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    print(f"Test Accuracy: {correct/total:.4f}\n")

if __name__ == "__main__":
    train()
