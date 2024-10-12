import torch

def get_teacher_probs(self, teacher_output):
    # print(torch.cuda.device_count())
    knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device='cuda:0')
    # knn_prob = knn_prob.to(self.teacher_model.device)
    combined_prob, _ = self.combiner.get_combined_prob(knn_prob, teacher_output[0] / self.prior_tau,
                                                       log_probs=False)
    # distil_lprobs = combined_prob
    distil_lprobs = combined_prob.view(-1, combined_prob.size(-1))
    return distil_lprobs

def get_knn_prob(self, vals, distances, temperature=None, device="cuda:0", **kwargs):

    temperature = temperature if temperature is not None else self.temperature
    return calculate_knn_prob(vals, distances, self.probability_dim,
                 temperature, device, **kwargs)

def get_combined_prob(self, knn_prob, neural_model_logit, lambda_ = None, log_probs = False):
    lambda_ = lambda_ if lambda_ is not None else self.lambda_
    return calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs)


def calculate_knn_prob(vals, distances, probability_dim, temperature, device, **kwargs):
    scaled_dists = - distances / temperature
    knn_weights = torch.softmax(scaled_dists, dim=-1)
    B, S, K = vals.size()
    # construct prob
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)
    return knn_probs

def calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs):
    r"""
    How vanilla knn-mt calculate the combining probability.
    """
    neural_model_prob = F.softmax(neural_model_logit, dim=-1)
    combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)

    # some extra infomation
    extra = {}
    extra["neural_probs"] = neural_model_prob
    extra["unlog_combined_probs"] = combined_probs

    if log_probs:
        combined_probs =  torch.log(combined_probs)
    return combined_probs, extra

distil_lprobs = self.get_teacher_probs(net_output)
com_prob = torch.log(distil_lprobs)
golden_loss, nll_loss = label_smoothed_nll_loss(com_prob, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,)
loss = golden_loss

