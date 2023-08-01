from torch.utils.data import Dataset

class PatientDataset(Dataset):
    def __init__(self, slides_dir, patient_ids, gene_expression_file, transform=None, color_normalization=None):
        self.slides_dir = slides_dir
        self.gene_expression_data = pd.read_csv(gene_expression_file, index_col='PATIENT_ID')
        # Apply the log2(1+x) transformation to the entire gene_expression_data DataFrame
        self.gene_expression_data = np.log2(1 + self.gene_expression_data)

        self.patient_ids = patient_ids
        
        self.model_transform = transform 
        
        self.color_norm = color_normalization
        
        self.preprocess = transforms.Compose([
                self.color_norm,
                self.model_transform,
                #transforms.ToTensor(),
                #transforms.Resize((224, 224)),
                
        ])
        
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]

        img_paths = sorted(glob.glob(os.path.join(self.slides_dir, patient_id, '*.png')))
        
        images = []
        for img_path in img_paths:
            img = Image.open(img_path)
            img = self.preprocess(img)
            images.append(img)
        
        images = torch.stack(images)

        # load the gene expression data for this patient
        gene_expression = self.gene_expression_data.loc[patient_id]
        gene_expression = torch.tensor(gene_expression.values, dtype=torch.float32)

        return images, gene_expression
