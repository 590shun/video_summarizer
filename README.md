# Video Summarization

## Installation
First, please clone this repository.  
```
git clone https://github.com/590shun/summarizer.git
```
Next, you will need to install the necessary packages.  
```
pip install -r requirements.txt
```

## Dataset Preparation
If you use public datasets(HDF5 format), run `datasets/download_datasets.py`.  

The videos can be downloaded from the following link.  

[TVSum(Video, 641MB)](http://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz)  
[SumMe(Video, 2.2GB)](https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip)  

Follow the steps below to set up the dataset.
```
cd datasets/
wget https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip
unzip SumMe.zip
```

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
You can also evaluate in ways other than f1 score.  
For more information, see [Kendall](https://academic.oup.com/biomet/article-pdf/33/3/239/573257/33-3-239.pdf) and [Spearman](http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Probability%20and%20statistics/CRC%20-%20standard%20probability%20and%20Statistics%20tables%20and%20formulae%20-%20DANIEL%20ZWILLINGER.pdf).
```
python evaluation.py
```

## Generate video summaries
When training, by the end of the classification, scores are computed on every video of the dataset using the best found weights. The generated summaries are saved in `logs/<timestamp>_<model_name>/<dataset_name>_preds.h5`. Video summaries can be generated using `summary.py`:
```
python summary.py -p logs/<timestamp>_<model_trainer_name>/<dataset_name>_splits.json_preds.h5 -f datasets/videos/summe/frames/Air_Force_One -d summarizer_dataset_summe_google_pool5.h5 -v video_1
```

## References
Yale Song et al. "TVSum: Summarizing web videos using titles", CVPR2015. [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf)  
Michael Gygli et al. "Creating Summaries from User Videos", ECCV2014. [paper](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_33)  
Jiri Fajtl et al. "Summarizing Videos with Attention", arXiv. [paper](http://arxiv.org/pdf/1812.01969)   
Behrooz Mahasseni et al. "Unsupervised Video Summarization with Adversarial LSTM Networks", CVPR2017. [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.pdf)  
Kaiyang Zhou et al. "Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward", AAAI2018. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16395/16358)  
Mayu Otani et al. "Rethinking the Evaluation of Video Summaries", CVPR2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Otani_Rethinking_the_Evaluation_of_Video_Summaries_CVPR_2019_paper.pdf)  


