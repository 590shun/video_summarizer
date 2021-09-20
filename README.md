# Video Summarization

## Installation
First, please clone this repository.  
```
git clone https://github.com/590shun/video_summarization.git
```
Next, you will need to install the necessary packages.  
```
pip install -r requirements.txt
```

## Dataset Preparation
If you use public datasets, run `datasets/download_datasets.py`.  
Alternatively, you can create your own dataset. In that case, run `generate_dataset.py`. For video feature extraction, see [here](https://github.com/590shun/Video-Feature-Extraction).

## Generate splits
```
python create_split.py -d datasets/summarizer_dataset_<dataset_name>_google_pool5.h5 --save-dir splits --save-name <dataset_name>_splits --num-splits 5
```

## Training
```
python main.py --model <model_name>
```
See `python main.py --help` for more parameters. Weights and logs are saved in `logs/<timestamp>_<model_name>`.


## Evaluation(Kendall's and Spearman's Correlation)


## Generate video summaries
When training, by the end of the classification, scores are computed on every video of the dataset using the best found weights. The generated summaries are saved in `logs/<timestamp>_<model_name>/<dataset_name>_preds.h5`. Video summaries can be generated using `summary.py`:
```
python summary.py -p logs/<timestamp>_<model trainer name>/<dataset name>_splits.json_preds.h5 -f datasets/videos/summe/frames/Air_Force_One -d summarizer_dataset_summe_google_pool5.h5 -v video_1
```



