import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph, PriorDiscriminator
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print("USER", args.user, "ITEM", args.item)
        print("NUM OF INTERACTIONS", self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ["Loss", "preLoss", "Recall", "NDCG"]
        for met in mets:
            self.metrics["Train" + met] = list()
            self.metrics["Test" + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = "Epoch %d/%d, %s: " % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += "%s = %.4f, " % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + "  "
        return ret

    def run(self):
        self.prepareModel()
        log("Model Prepared")
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics["TrainLoss"]) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log("Model Initialized")
        bestRes = None
        for ep in range(stloc, args.epoch):
            tstFlag = ep % args.tstEpoch == 0
            reses = self.trainEpoch()
            log(self.makePrint("Train", ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                log(self.makePrint("Test", ep, reses, tstFlag))
                self.saveHistory()
                bestRes = (
                    reses
                    if bestRes is None or reses["Recall"] > bestRes["Recall"]
                    else bestRes
                )
            print()
        reses = self.testEpoch()
        log(self.makePrint("Test", args.epoch, reses, True))
        log(self.makePrint("Best Result", args.epoch, bestRes, True))
        self.saveHistory()

    def prepareModel(self):
        self.model = Model()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs()
        self.sampler = LocalGraph()
        self.discriminator = PriorDiscriminator(args.latdim)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader

        # -----
        A = self.handler.torchBiAdj
        idxs = A._indices()
        vals = t.where(A._values() > 0, 1, 0)
        shape = A.shape
        L = t.sparse.FloatTensor(idxs, vals, shape)

        # dense_eye = t.eye(A.size(0))
        # sparse_eye = t.sparse.FloatTensor(dense_eye)
        # A_pseudo_inv = dense_eye / (A @ sparse_eye).to_dense()
        A_pseudo_inv = t.where(L._values() > 0, -1, 1)
        A_pseudo_inv = t.sparse.FloatTensor(idxs, A_pseudo_inv, shape)
        A_adv = A + A_pseudo_inv * L
        # print("A_adv", A_adv)
        bce = t.nn.BCELoss()
        # -----

        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, tem in enumerate(trnLoader):
            if i % args.fixSteps == 0:
                sampScores, seeds = self.sampler(
                    self.handler.allOneAdj, self.model.getEgoEmbeds()
                )

                encoderAdj, decoderAdj = self.masker(self.handler.torchBiAdj, seeds)

                adv_encoderAdj, adv_decoderAdj = self.masker(A_adv, seeds)

            ancs, poss, _ = tem
            ancs = ancs.long()
            poss = poss.long()
            usrEmbeds, itmEmbeds = self.model(encoderAdj, decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]

            adv_usrEmbeds, adv_itmEmbeds = self.model(adv_encoderAdj, adv_decoderAdj)

            D1u = self.discriminator(usrEmbeds)
            D1i = self.discriminator(itmEmbeds)
            D2u = self.discriminator(adv_usrEmbeds)
            D2i = self.discriminator(adv_itmEmbeds)
            D1u_label_adv_target = t.FloatTensor(D1u.data.size()).fill_(1)
            D1i_label_adv_target = t.FloatTensor(D1i.data.size()).fill_(1)
            D2u_label_adv_target = t.FloatTensor(D2u.data.size()).fill_(0)
            D2i_label_adv_target = t.FloatTensor(D2i.data.size()).fill_(0)
            ploss = (
                bce(D1u, D1u_label_adv_target)
                + bce(D1i, D1i_label_adv_target)
                + bce(D2u, D2u_label_adv_target)
                + bce(D2i, D2i_label_adv_target)
            )

            bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            regLoss = calcRegLoss(self.model) * args.reg

            contrastLoss = (
                contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)
            ) * args.ssl_reg + contrast(ancs, usrEmbeds, itmEmbeds)

            loss_weights = t.softmax(t.tensor([-bprLoss, -regLoss, -contrastLoss, -ploss]), 0)
            alpha1 = 0.4 + 0.01 * loss_weights[0]
            alpha2 = 0.4 + 0.01 * loss_weights[1]
            alpha3 = 0.8 + 0.01 * loss_weights[2]
            alpha4 = 4 - alpha1 - alpha2 - alpha3 + 0.01 * loss_weights[3]
           
            loss = (
                alpha1 * bprLoss
                + alpha2 * regLoss
                + alpha3 * contrastLoss
                + alpha4 * ploss
            )

            if i % args.fixSteps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log(
                "Step %d/%d: loss = %.1f, reg = %.1f, cl = %.1f   "
                % (i, steps, loss, regLoss, contrastLoss),
                save=False,
                oneline=True,
            )
        ret = dict()
        ret["Loss"] = epLoss / steps
        ret["preLoss"] = epPreLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall, epNdcg = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long()

            usrEmbeds, itmEmbeds = self.model(
                self.handler.torchBiAdj, self.handler.torchBiAdj
            )

            allPreds = (
                t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask)
                - trnMask * 1e8
            )
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(
                topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr
            )
            epRecall += recall
            epNdcg += ndcg
            log(
                "Steps %d/%d: recall = %.1f, ndcg = %.1f          "
                % (i, steps, recall, ndcg),
                save=False,
                oneline=True,
            )
        ret = dict()
        ret["Recall"] = epRecall / num
        ret["NDCG"] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum(
                [
                    np.reciprocal(np.log2(loc + 2))
                    for loc in range(min(tstNum, args.topk))
                ]
            )
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open("./History/" + args.save_path + ".his", "wb") as fs:
            pickle.dump(self.metrics, fs)

        content = {
            "model": self.model,
        }
        t.save(content, "./Models/" + args.save_path + ".mod")
        log("Model Saved: %s" % args.save_path)

    def loadModel(self):
        ckp = t.load("./Models/" + args.load_model + ".mod")
        self.model = ckp["model"]
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open("./History/" + args.load_model + ".his", "rb") as fs:
            self.metrics = pickle.load(fs)
        log("Model Loaded")


if __name__ == "__main__":
    logger.saveDefault = True

    log("Start")
    handler = DataHandler()
    handler.LoadData()
    log("Load Data")

    coach = Coach(handler)
    coach.run()
