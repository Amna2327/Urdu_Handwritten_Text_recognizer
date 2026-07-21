import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from dataset import my_Dataset, Vocab_builder, collate_fn
from models.v1_CNN_BiLSTM import v1_Recognizer
from transforms import transform, train_transform


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    train_loss = 0
    c = 0
    for image_batch, labels, label_lengths, input_length in train_loader:
        optimizer.zero_grad()
        c += 1

        image_batch = image_batch.to(device)
        labels = labels.to(device)
        input_length = input_length.to(device)

        with autocast('cuda'):
            log_probs = model(image_batch)
            log_probs = log_probs.transpose(0, 1)
            # CTC loss kept in float32 for numerical stability even inside autocast
            loss = criterion(log_probs.float(), labels, input_length, label_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        if c % 100 == 0:
            print(f"Batch number: {c}, loss(per batch): {loss.item()}")

    return train_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    for image_batch, labels, label_lengths, input_length in val_loader:
        with torch.no_grad():
            image_batch = image_batch.to(device)
            labels = labels.to(device)
            input_length = input_length.to(device)

            with autocast('cuda'):
                log_probs = model(image_batch)
                log_probs = log_probs.transpose(0, 1)
                loss = criterion(log_probs.float(), labels, input_length, label_lengths)

            total_loss += loss.item()
    return total_loss


def main():
    # Fixed input size (always padded to 1400 width) -> let cuDNN cache the
    # fastest conv algorithms for this exact shape instead of re-searching.
    torch.backends.cudnn.benchmark = True

    word2Index, Index2word = Vocab_builder("data/DataSet/UHWR/train.txt")

    train_Dataset = my_Dataset(
        label_path="data/DataSet/UHWR/train.txt",
        word2Index=word2Index,
        Index2word=Index2word,
        dataset_path="data/DataSet/UHWR/",
        transform=train_transform
    )

    val_Dataset = my_Dataset(
        label_path="data/DataSet/UHWR/val.txt",
        word2Index=word2Index,
        Index2word=Index2word,
        dataset_path="data/DataSet/UHWR/",
        transform=transform
    )

    # num_workers lets CPU augment the next batch while GPU trains on the current one.
    # persistent_workers=True avoids respawning worker processes every single epoch.
    train_loader = DataLoader(
        train_Dataset, batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_fn(b, word2Index),
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_Dataset, batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_fn(b, word2Index),
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    model = v1_Recognizer()
    criterion = torch.nn.CTCLoss(blank=117, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler('cuda')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    epochs = 100
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 8

    for i in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)

        # scheduler reduces LR if val_loss stalls
        scheduler.step(avg_val_loss)

        print(f'epoch: {i+1}, train_loss= {avg_train_loss:.4f}, val_loss= {avg_val_loss:.4f}')

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/best_model_3layer_BiLSTM_augmented.pt")
            print(f'  -> new best val_loss, saved checkpoint')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {i+1}')
                break


if __name__ == "__main__":
    main()