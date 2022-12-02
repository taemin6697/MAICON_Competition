"""
Predict
"""
from tqdm import tqdm
import numpy as np
import random, os, sys, torch, cv2, warnings
from glob import glob
from torch.utils.data import DataLoader
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from modules.utils import load_yaml, save_yaml, get_logger
import datasets
warnings.filterwarnings('ignore')

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)
PRETRAINED = 'nvidia/segformer-b4-finetuned-cityscapes-1024-1024'

feature_extractor = SegformerFeatureExtractor.from_pretrained(PRETRAINED,reduce_labels=False)

def test_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    filenames = [x.split("/")[-1] for x in example_batch['filenames']]
    orig_size = [cv2.imread(x, cv2.IMREAD_COLOR).shape for x in example_batch['filenames']]
    inputs = feature_extractor(images,return_tensors="pt")
    inputs['filename'] = filenames
    inputs['orig_size'] = orig_size
    return inputs

if __name__ == '__main__':

    #! Load config
    config = load_yaml(os.path.join(prj_dir, 'config', 'predict.yaml'))
    train_config = load_yaml(os.path.join(prj_dir, 'config', 'train.yaml'))

    #! Set predict folder name
    pred_folder = config['train_folder']

    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create train result directory and set logger
    pred_result_dir = os.path.join(prj_dir, 'results', 'pred', pred_folder)
    pred_result_dir_mask = os.path.join(prj_dir, 'results', 'pred', pred_folder, 'mask')
    os.makedirs(pred_result_dir, exist_ok=True)
    os.makedirs(pred_result_dir_mask, exist_ok=True)

    # Set logger
    logging_level = 'debug' if config['verbose'] else 'info'
    logger = get_logger(name='train',
                        file_path=os.path.join(pred_result_dir, 'pred.log'),
                        level=logging_level)

    # Set data directory
    test_dirs = os.path.join(prj_dir, 'data', 'test')
    test_img_paths = glob(os.path.join(test_dirs, 'x', '*.png'))
    
    test_dataset = datasets.Dataset.from_dict(
            {"image": test_img_paths,
             "filenames":list(map(lambda x:x, test_img_paths))},
            features=datasets.Features({
                "image": datasets.Image(),
                "filenames":datasets.Value(dtype='string', id=None),
                }))
    test_dataset.set_transform(test_transforms)

    # Create data loader
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                shuffle=False,
                                drop_last=False)
    logger.info(f"Load test dataset: {test_dataset}")

    # Load architecture
    check_point_path = glob(os.path.join(prj_dir, 'results', 'train', config['train_folder'], 'checkpoint-*'))
    check_point_path = sorted(check_point_path)[0]

    model = SegformerForSemanticSegmentation.from_pretrained(check_point_path).to(device)

    # Save config
    save_yaml(os.path.join(pred_result_dir, 'train_config.yml'), train_config)
    save_yaml(os.path.join(pred_result_dir, 'predict_config.yml'), config)

    # Predict
    logger.info(f"START PREDICTION")

    model.eval()

    with torch.no_grad():
        for batch_id, x in enumerate(tqdm(test_dataloader)):
            y_pred = model(x['pixel_values'].to(device))
            logits = y_pred.logits
            y_pred = logits.detach().cpu().numpy()

            y_pred_argmax = y_pred.argmax(1).astype(np.uint8)

            orig_size = [(
                x['orig_size'][0].tolist()[i], 
                x['orig_size'][1].tolist()[i]) for i in range(len(x['orig_size'][0]))]

            # Save predict result
            for filename_, orig_size_, y_pred_ in zip(x['filename'], orig_size, y_pred_argmax):
                resized_img = cv2.resize(y_pred_, [orig_size_[1], orig_size_[0]], interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(pred_result_dir_mask, filename_), resized_img)
    logger.info(f"END PREDICTION")