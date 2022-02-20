from sentence_transformers.util import cos_sim
from preprocessing.helpers import row2zero, col2zero

#fast clustering implementation taken directly from sentence_transformers

try:
    from sentence_transformers.util import community_detection

except:
    from sentence_transformers.util import cos_sim

    def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):
        """
        Function for Fast Community Detection
        Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
        Returns only communities that are larger than min_community_size. The communities are returned
        in decreasing order. The first element in each list is the central point in the community.
        """

        # Maximum size for community
        init_max_size = min(init_max_size, len(embeddings))

        # Compute cosine similarity scores
        cos_scores = cos_sim(embeddings, embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        extracted_communities = []
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
                top_idx_large = top_idx_large.tolist()
                top_val_large = top_val_large.tolist()

                if top_val_large[-1] < threshold:
                    for idx, val in zip(top_idx_large, top_val_large):
                        if val < threshold:
                            break

                        new_cluster.append(idx)
                else:
                    # Iterate over all entries (slow)
                    for idx, val in enumerate(cos_scores[i].tolist()):
                        if val >= threshold:
                            new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        # Largest cluster first
        extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

        # Step 2) Remove overlapping communities
        unique_communities = []
        extracted_ids = set()

        for community in extracted_communities:
            add_cluster = True
            for idx in community:
                if idx in extracted_ids:
                    add_cluster = False
                    break

            if add_cluster:
                unique_communities.append(community)
                for idx in community:
                    extracted_ids.add(idx)

        return unique_communities

def form_similarity_matrix(embeddings):    
    return cos_sim(embeddings, embeddings)

def similar_object_grouping(embeddings, threshold, group_size):

    similarity_matrix = form_similarity_matrix(embeddings=embeddings)
    aux_similarity_matrix = similarity_matrix

    num_objects = len(embeddings)
    unique_groups = []

    for row in range(num_objects):
        
        group = []
        for col in range(num_objects):

            #if similarity (rounded) is >= threshold, objects are similar
            if round(float(aux_similarity_matrix[row][col]), 1) >= threshold:
                group.append(col)

                #set similarity of this object string with every other object as zero
                aux_similarity_matrix = row2zero(aux_similarity_matrix, col)
                aux_similarity_matrix = col2zero(aux_similarity_matrix, col)

        #only those groups with members >= minimum group size are important unique groups for us
        if len(group) >= group_size:
            unique_groups.append(group)

    return unique_groups

def map_index_to_unique_objects(unique_objects, grouped_indices):

    grouped_objects = []

    #convert each group of indices to group of object strings
    for group in grouped_indices:

        grouped_together = []

        #iterate through the ids of each group and map them to the respective index of unique_objects
        for idx in group:

            grouped_together.append(unique_objects[idx])

        grouped_objects.append(grouped_together)

    return grouped_objects

def similar_object_communities(unique_objects, embeddings, threshold, min_community_size):

    unique_communities = community_detection(embeddings=embeddings, threshold=threshold, min_community_size=min_community_size)

    object_groups = map_index_to_unique_objects(unique_objects=unique_objects, grouped_indices=unique_communities)

    return object_groups

def similar_object_detection(unique_objects, embeddings, threshold, min_group_size):

    similar_object_groups = similar_object_grouping(embeddings=embeddings, threshold=threshold, group_size=min_group_size)

    object_groups = map_index_to_unique_objects(unique_objects=unique_objects, grouped_indices=similar_object_groups)

    return object_groups