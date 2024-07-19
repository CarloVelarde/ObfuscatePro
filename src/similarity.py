import torch
import torch.nn.functional as F

def calculate_cosine_similarity(embedding1, embedding2):
    vec1 = torch.tensor(embedding1, dtype=torch.float32)
    vec2 = torch.tensor(embedding2, dtype=torch.float32)

    vec1 = vec1.unsqueeze(0)
    vec2 = vec2.unsqueeze(0)

    cosine_sim = F.cosine_similarity(vec1, vec2)
    return cosine_sim.item()


def calculate_euclidean_distance(embedding1, embedding2):
    vec1 = torch.tensor(embedding1, dtype=torch.float32)
    vec2 = torch.tensor(embedding2, dtype=torch.float32)

    

    euclidean_distance = torch.norm(vec1 - vec2, p= 2)
    return euclidean_distance.item()