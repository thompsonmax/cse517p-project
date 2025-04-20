# Checkpoint 1

## Dataset

At the time of writing, we plan to use the "common corpus" dataset [0] hosted on HuggingFace. This is a very large dataset (2 trillion tokens) focused on permissively licensed content with documented provenance. The goal of the dataset curators is to provide a high-quality and ethically source dataset. We very likely will not use all of the text in the dataset and will perform stratified sampling per language to try to get good coverage over many different languages. This dataset has over 1 billion tokens each for over 30 languages.

We will download the data directly from the HuggingFace datasets website [0]. The site allows for downloading a subset of the data at once. We will make use of this functionality to experiment with a subset of the data and then scale up once we achieve good results with our chosen model. The data can be downloaded in Parquet format, which we can load directly into Pandas or Google BigQuery for data exploration and refinement.

If this dataset is insufficient to cover all the languages we want to support, we may explore mixing in data from these additional sources:

* Common Crawl - https://commoncrawl.org/
* Wikipedia Dumps - https://dumps.wikimedia.org/
* ParaCrawl - https://paracrawl.eu/

[0] https://huggingface.co/datasets/PleIAs/common_corpus

## Method

<!--- TODO: MORE CONTENT HERE -->

### Model

We plan to use a Transformer-based model architecture, which currently is what is commonly used to achieve state of the art performance along the Pareto frontier. Particularly we will evaluate a few state of the art approaches such as Transformer-XL and LongFormer. We plan to train and run the model within pytorch.

### Encoding

Since the input characters are UTF-8 encoded, we will need to carefully consider how we craft the input and output of the model. UTF-8 presents some additional difficulty as characters are encoded using variable length sequences of 1-4 bytes per character. We will transform the UTF-8 characters into fixed size unicode code points as the input to the model. Similarly, the output encoding of the model will a unicode code point.

Since the total number of code points in unicode is vast, we will focus on supporting only the Basic Multilingual Plane (BMP) of code points that range from U+0000 to U+FFFF (65535 code points in total). This will allow us to support the basic character set required to support almost every langauage without making the vector encodings for each character too huge and thus compromising the speed and efficiency of the model. For rare characters that we encounter outside of the BMP, we will replace them with an '<UNK>' character in our encoding.

We may consider exploring UTF-8 byte level encoding over directly encoding the unicode code points if that potentially can increase performance. However, that introduces additional complexity on processing the input and running inference. Particular with inference, we would need to be able to detect when we need to run the model again to generate additional UTF-8 characters to produce a valid code point. So for now we will stick with directly encoding the unicode code points.

#### Input

On top of the conversion to unicode code points above, we will need to additionally define how these code point sequences get encoded as input into the model. 

#### Output

We need to get the top 3 characters from the output of the model when running predictions. We will first run our softmax layer on the output vectors to get a probability distribution over all characters. We will then use `pytorch.argsort` to get the indices corresponding to the top 3 probability values in the vector.