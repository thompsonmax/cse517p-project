# Checkpoint 1

## Dataset

At the time of writing, we plan to use the "common corpus" dataset [0] hosted on HuggingFace. This is a very large dataset (2 trillion tokens) focused on permissively licensed content with documented provenance. The goal of the dataset curators is to provide a high-quality and ethically source dataset. We very likely will not use all of the text in the dataset and will perform stratified sampling per language to try to get good coverage over many different languages. This dataset has over 1 billion tokens each for over 30 languages.

We will download the data directly from the HuggingFace datasets website [0]. The site allows for downloading a subset of the data at once. We will make use of this functionality to experiment with a subset of the data and then scale up once we achieve good results with our chosen model.

If this dataset is insufficient to cover all the languages we want to support, we may explore mixing in data from these additional sources:

* Common Crawl - https://commoncrawl.org/
* Wikipedia Dumps - https://dumps.wikimedia.org/
* ParaCrawl - https://paracrawl.eu/

[0] https://huggingface.co/datasets/PleIAs/common_corpus

## Method

<!--- TODO: MORE CONTENT HERE -->

### Model

We plan to use a Transformer-based model architecture, which currently is what is commonly used to achieve state of the art performance along the Pareto frontier. Particularly we will evaluate a few state of the art approaches such as Transformer-XL and LongFormer. We plan to train and run the model within pytorch.

### Input and Output formats

Since the input characters are UTF-8 encoded, we will need to carefully consider how we craft the input and output of the model.