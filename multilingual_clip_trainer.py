import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, CLIPVisionModel
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

class MultilingualCLIP(nn.Module):
    def __init__(self, text_model_name, image_model_name, projection_dim):
        super(MultilingualCLIP, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        if image_model_name:
            self.image_encoder = CLIPVisionModel.from_pretrained(image_model_name)
            in_features = self.image_encoder.config.hidden_size
            self.image_projection = nn.Linear(in_features, projection_dim)
        else:
            self.image_encoder = None
            self.image_projection = None
        
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        
    def forward(self, input_ids, attention_mask, pixel_values=None):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_features)
        
        if pixel_values is not None and self.image_encoder is not None:
            image_features = self.image_encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            image_embeddings = self.image_projection(image_features)
            return text_embeddings, image_embeddings
        
        return text_embeddings

class MultilingualTextDataset(Dataset):
    def __init__(self, captions, tokenizer, max_length):
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        encoding = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def train_multilingual_clip(model, train_loader, optimizer, device, epochs, use_images):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            if use_images:
                pixel_values = batch['pixel_values'].to(device)
                text_embeddings, image_embeddings = model(input_ids, attention_mask, pixel_values)
                logits = torch.matmul(text_embeddings, image_embeddings.t())
            else:
                text_embeddings = model(input_ids, attention_mask)
                logits = torch.matmul(text_embeddings, text_embeddings.t())
            
            labels = torch.arange(logits.shape[0]).to(device)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch)

def main():
    mlflow.set_experiment("Multilingual CLIP Training")
    
    with mlflow.start_run():
        # Set parameters
        text_model_name = 'BAAI/bge-m3'
        image_model_name = 'openai/clip-vit-base-patch32'  # Set to None if not using images
        projection_dim = 512
        max_length = 128
        batch_size = 32
        learning_rate = 1e-4
        epochs = 10
        use_images = False  # Set to True if using images
        
        # Log parameters
        mlflow.log_params({
            "text_model": text_model_name,
            "image_model": image_model_name,
            "projection_dim": projection_dim,
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "use_images": use_images
        })
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = MultilingualCLIP(text_model_name, image_model_name, projection_dim).to(device)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Prepare dataset and dataloader
        captions = [...]  # List of multilingual captions
        dataset = MultilingualTextDataset(captions, tokenizer, max_length)
        
        if use_images:
            # You'll need to implement image loading and preprocessing
            # Modify dataset to include images if use_images is True
            pass
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        train_multilingual_clip(model, train_loader, optimizer, device, epochs, use_images)
        
        # Save the trained model
        mlflow.pytorch.log_model(model, "model")

if __name__ == '__main__':
    main()