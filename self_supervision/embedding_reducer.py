import umap

class EmbeddingReducer:
    """
        Class to convert high dimensional embeddings into lower dimensional embeddings.
        Current implementation uses only UMAP to do so

        @TODO:
            - add t-SNE mode
            - add isomap mode
            - add locally linear embedding mode
    """

    def __init__(self, embedding_to_reduce, dimension=None, mode=None, optimal_size=10):
        """
        Initialize parameters for dimensionality reduction model

        Args:
            embedding_to_reduce (list | torch.Tensor | np.array): list | tensor | array of high dimensional embeddings
            dimension (int, optional): output number of dimensions. Defaults to None.
            optimal_size (int, optional): number of points to look at to form a local representation. Larger value makes less fine grained distinctions in data. Defaults to 10.
        """
        
        self.embeddings = embedding_to_reduce
        self.dimensions = dimension
        self.mode = mode
        self.optimal_size = optimal_size

        if self.dimensions == None:
            self.dimensions = 2

    def umap_dimensionality_reduction(self):
        return umap.UMAP(n_components=self.dimensions,
                        n_neighbors=3*self.optimal_size,
                        min_dist=0.0,
                        metric='cosine',
                        random_state=0,
                        low_memory=False)

    def get_output(self):

        mode =  self.mode

        if mode == None or mode == "umap":
            self.dimension_reduction_model = self.umap_dimensionality_reduction()
            
        # print(type(self.embeddings))
        reduced_embeddings = self.dimension_reduction_model.fit_transform(X=self.embeddings)
        
        return reduced_embeddings