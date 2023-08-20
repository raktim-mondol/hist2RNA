# Import Libraries
from libs import *


class PatientDataset(Dataset):
    def __init__(self, slides_dir, patient_ids, transform=None, color_normalization=None):
        self.slides_dir = slides_dir
        
        self.patient_ids = patient_ids
        
        self.model_transform = transform 
        
        self.color_norm = color_normalization
        
        self.preprocess = transforms.Compose([
                self.color_norm,
                self.model_transform               
        ])
        
    def __len__(self):
        return len(self.patient_ids)
        
    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        img_paths = sorted(glob.glob(os.path.join(self.slides_dir, patient_id, '*.png')))
        
        images = []
        for img_path in img_paths:
            try:
                img = Image.open(img_path)
                img = self.preprocess(img)
                images.append(img)
            except Exception as e:
                print(f"Failed to open or preprocess image at {img_path}. Error: {str(e)}")
        
        if len(images) == 0:
            raise RuntimeError(f"No images could be read for patient id {patient_id}.")
    
        images = torch.stack(images)
    
        return images, patient_id
        

# Add other necessary imports
BATCH_SIZE = 1

# Save directory for extracted features
FEATURE_SAVE_PATH = "./saved_features/resnet50/"

# Hyperparameters
BASE_MODEL_NAME = "resnet50"  # or other models

# Path to slides

slides_dir = "/data/raw_wsi_tcga_images/"

# Define patients or load patient ids
patient_list = pd.read_csv('./data/patient_details/all_patient_list.txt', index_col='all_patient_id')
patient_ids = patient_list.index.astype(str).tolist()

# Define the base model and transformation
def get_base_model_and_transforms(model_name):
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif model_name == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT
        model = efficientnet_v2_m(weights=weights)  
        num_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        
    elif model_name == "vit_l_32":
        weights = ViT_L_32_Weights.DEFAULT
        model = vit_l_32(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        
    elif model_name == "vit_l_16":
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        
    elif model_name == "maxvit_t":
        weights = MaxVit_T_Weights.DEFAULT
        model = maxvit_t(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],  # AdaptiveAvgPool2d(output_size=1)
            model.classifier[1]  # Flatten(start_dim=1, end_dim=-1)
            )
    elif model_name == "swin_b":
        weights = Swin_B_Weights.DEFAULT
        model = swin_b(weights=weights)
        num_features = model.head.in_features
        model.head = nn.Identity()
        
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # Make all the layer of the base model trainable 
    for param in model.parameters():
        param.requires_grad = False
    
    preprocess = weights.transforms()
    
    # model name; number of output features, and model specific preprocess
    return model, num_features, preprocess


def collate_fn(batch):
    images, patient_id = zip(*batch)
    clipped_images = []
    for img in images:
        if len(img) > 1000:
            img = img[:1000]  # If more than 1000 patches, clip it to 1000
        clipped_images.append(img)
    return clipped_images, patient_id


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model, num_features, preprocess = get_base_model_and_transforms(BASE_MODEL_NAME)

color_normalization = MacenkoColorNormalization()

dataset = PatientDataset(slides_dir, patient_ids, transform=preprocess, color_normalization=color_normalization)

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# Main code to extract features
if __name__ == "__main__":
    base_model, _, preprocess = get_base_model_and_transforms(BASE_MODEL_NAME)
    base_model.eval()
    if torch.cuda.device_count() > 1:
        print("Multiple GPU Detected")
        print(f"Using {torch.cuda.device_count()} GPUs")
        base_model = nn.DataParallel(base_model)
    base_model.to(device)
    with torch.no_grad():
        for images, patient_id in data_loader:
            #print(images)
            images = tuple(batch.to(device) for batch in images)
            for i, patient_images in enumerate(images):
                pid = patient_id[i]
                print(pid)
                print(patient_images.shape)
                patient_features = base_model(patient_images)
                # Aggregated features 
                aggregated_feature = patient_features.mean(dim=0)
                aggregated_feature=aggregated_feature.unsqueeze(0)
                torch.save(aggregated_feature, f"{FEATURE_SAVE_PATH}{pid}.pt")
                print(f"Saved features for patient {pid}")
