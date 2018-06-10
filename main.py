"""Run at terminal with --train flag to train Doc2Vec model with Gensim
Run at terminal with --visualise flag to visualise Doc2Vec vectors with hypertools."""
import jsonlines, time, argparse, pickle
import pandas as pd
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import hypertools as hyp
import seaborn as sns
attributes = {
    'axes.facecolor' : '#f0e6f2'
}
sns.set(context='paper',style='darkgrid',rc=attributes)

parser = argparse.ArgumentParser(description="Train gensim's Doc2Vec with 1m articles.")
parser.add_argument('--train', action='store_true')
parser.add_argument('--visualise', action='store_true')


class IterDocs():
    """Iterator to stream articles to Doc2Vec model"""
    def __init__(self):
        super(IterDocs, self).__init__()
        self.tags = np.array([])
        self.metadata = pd.DataFrame(columns=['id', 'media-type', 'published', 'source', 'title'])
        self.num_articles = 5000

    def __iter__(self):
        """Stream articles from sample file and output tokenised text and ID"""
        with jsonlines.open('data/sample-1M.jsonl') as reader:
            for counter,article in enumerate(reader):
                if counter == self.num_articles:
                    break
                doc = gensim.utils.simple_preprocess(article['content'])
                tag = article['id']
                self.tags = np.append(self.tags, tag)
                self.metadata.loc[counter] = [article['id'], article['media-type'], article['published'], article['source'], article['title']]
                yield TaggedDocument(words=doc, tags=[tag])

class Analysis():
    def __init__(self):
        super(Analysis, self).__init__()
        self.docs = IterDocs()

    def initialise(self):
        """Initialise doc2vec model"""
        print('Initialise model')
        start_time = time.time()

        self.model = Doc2Vec(self.docs, vector_size=100, window=8, min_count=5, workers=4, epochs=10) # train_words=False?

        print('Model initialised in', time.time() - start_time)

    def train(self):
        print('Begin training')
        start_time = time.time()
        
        self.model.train(self.docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
        print('Trained in:', time.time() - start_time)

    def save(self):
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.model.save('data/trained.bin')
        print('Model saved to data/trained.bin')

        # pickle metadata
        self.docs.metadata.to_csv('metadata.csv')


def training_wrapper():
    analysis = Analysis()
    analysis.initialise()
    analysis.train()
    analysis.save()
    print(analysis.docs.tags[0])
    print(analysis.model.docvecs[analysis.docs.tags[0]].round(2))


def load_viz_data():
    """ Return loaded data:
    metadata - pandas dataframe containing article metadata
    tags     - dict with article ids as keys and corresponding index in vectors as value
    vectors  - array of Doc2Vec article vectors
    """

    print('Loading model')
    model = Doc2Vec.load('data/trained.bin')
    print('Model loaded')

    # load metadata
    metadata = pd.read_csv('metadata.csv')
    # print(metadata)

    tags = model.docvecs.doctags
    vectors = model.docvecs.vectors_docs
    return metadata, tags, vectors

def plot_pca_matplotlib(metadata, vectors):
    """Written before switched to hypertools"""
    # do PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    # plot PCA vectors
    fig, ax = plt.subplots()
    ax.grid = False

    cmap = {'News' : 'b', 'Blog' : 'g'}
    cmapped = metadata.replace({'media-type': cmap})
    ax.scatter(result[:,0], result[:,1], c=cmapped['media-type'], s=8)#, color='purple')
    for index, row in metadata.iterrows():
        ax.annotate(row.title, (result[index,0], result[index,1]))
    plt.show()

def viz_wrapper():
    metadata, tags, vectors = load_viz_data()
    print(vectors[tags['1aa9d1b0-e6ba-4a48-ad0c-66552d896aac'][0]])

    # pick dimensionality reduction technique and method

    # plot_pca_matplotlib(metadata, vectors)
    geo = hyp.plot(vectors, '.', ndims=3)
    geo = hyp.plot(vectors, '.', ndims=3, reduce='TSNE', hue=metadata.source)
    geo = hyp.plot(vectors, '.', ndims=3, reduce='UMAP')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.train:
        training_wrapper()
    if args.visualise:
        viz_wrapper()