import torch
import torch.nn as nn
from torch.autograd import Variable

#from torch.autograd import Variable
from torch import nn, Tensor

class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        #ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        #an = torch.clamp_min(sn.detach() + self.m, min=0.)
        #print("self.gamma: ", self.gamma)
        #delta_p = 1 - self.m
        #delta_n = self.m

        #logit_p = - ap * (sp - delta_p) * self.gamma
        #logit_n = an * (sn - delta_n) * self.gamma
        logit_p = sp * self.gamma
        #print("logit_p: ", logit_p)
        logit_n = sn * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0))/self.gamma 
        loss1 = self.soft_plus(torch.logsumexp(logit_p, dim=0))/self.gamma

        return loss, loss1

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.m = 0.25
        self.gamma = 256
        self.criterion = CircleLoss(self.m, self.gamma)
        self.criterion2 = CircleLoss(self.m, 1)
        #self.circle_loss = criterion(inp_sp, inp_sn)

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    #def forward(self, im, s, s_l):

        #scores = compute_sim_score(im, s, self.opt)
    #    scores = cosine_similarity(im, s)
        
    #    diagonal = scores.diag().view(-1, 1)
    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        #print("diagonal: ", diagonal.size())
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1)#.clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2)#.clamp(min=0)
        #sp = diagonal.view(im.size(0))
        #print("sp: ", sp.size())
        #sn = scores

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, float(-1000))#float('-inf')
        #inf = float('-inf')
        #sn = sn.masked_fill_(I, float(-1000))
        #sn = sn.view(im.size(0)*s.size(0))
        cost_im = cost_im.masked_fill_(I, float(-1000))

        cost_im = cost_im.t()
        circle_loss = 0

        middle = torch.zeros(d1.size(0)) * 1.0
        middle1 = torch.zeros(d1.size(0)) * 1.0
        #middle = Variable(middle_tem)
        #middle1 = Variable(middle1_tem)
        #if torch.cuda.is_available():
        #    middle = middle.cuda()
        #    middle1 = middle1.cuda()

        for i in range(d1.size(0)):
            middle[i], middle1[i] = self.criterion(cost_s[i], cost_im[i])#self.criterion(cost_s[i]*r_s[i], cost_im[i]*r_i[i])

        circle_loss1, circle_loss2 = self.criterion2(middle, middle1) # / 2.0
        #print("circle_loss: ", circle_loss)
        #kk = input()
        circle_loss = circle_loss1 + circle_loss2
        return circle_loss

class ContrastiveLoss_raw(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

