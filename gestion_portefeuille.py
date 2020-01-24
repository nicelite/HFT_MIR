# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:20:04 2017

@author: ARTHUR
"""

import matplotlib.pyplot as plt
from math import *
#from ARCH import *


def moyenne(L):
    '''calcul la moyenne d'une liste'''
    s = 0
    for n in L:
        n = float(n)
        s += n
    return s / len(L)


def volatilite(L):
    '''calcul la variance d'une liste'''
    T = len(L)
    mu = moyenne(L)
    t = 0
    s = 0
    while t < T:
        s += (L[t] - mu) * (L[t] - mu)
        t += 1
    return s / (T - 1)


def covariance(L1, L2):
    '''calcul la covariance entre deux listes'''
    T = len(L1)
    mu1 = moyenne(L1)
    mu2 = moyenne(L2)
    t = 0
    s = 0
    while t < T:
        s += (L1[t] - mu1) * (L2[t] - mu2)
        t += 1
    return s / (T - 1)


# gestion de portefeuille avec 2 titres


def gestion_de_portefeuille_2actions(indice, methode, graphique=True):
    '''utilise la théorie de la gestion de portefeuille et donne la répartition des 2 titres la moins risquée à appliquer à l'instant indice+1'''
    if methode == 'AIC':
        pmax = 10
    else:
        pmax = 11
    port_2actions = ARCH_2titres(r1, r2, indice, pmax, 15, 20, methode)
    inversible = port_2actions[4]
    if not inversible:
        return [0, 0, 0, 0]
    else:
        Action1 = port_2actions[0]
        Action2 = port_2actions[1]
        cov1_2 = covariance(port_2actions[2], port_2actions[3])
        Mu = []
        Sigma = []
        step = 0.01
        x1 = 0
        x2 = 1 - x1
        while x1 <= 1:
            mu = x1 * Action1[0] + x2 * Action2[0]
            sigma = (x1 ** 2) * Action1[1] + (x2 ** 2) * Action2[1] + 2 * x1 * x2 * cov1_2
            Mu.append(mu)
            Sigma.append(sigma)
            x1 += step
            x2 = 1 - x1
        sigma_min = min(Sigma)
        i = 0
        while Sigma[i] != sigma_min:
            i += 1

        couple_optimal = [Sigma[i], Mu[i]]  # coordonnées du point de risque minimum

        x1 = i * step
        x2 = 1 - x1
        if graphique:
            print("le couple optimal est:")
            print("mu=", couple_optimal[1])
            print("sigma=", couple_optimal[0])

            print("La proportion optimale d action 1 est=", x1)
            print("La proportion optimale d action 2 est=", x2)

            plt.plot(Action1[1], Action1[0], "g+", markersize=17, mew=2)
            plt.plot(Action2[1], Action2[0], "g+", markersize=17, mew=2)
            plt.plot(Sigma, Mu, "b+")
            plt.plot(couple_optimal[0], couple_optimal[1], 'r+', markersize=17, mew=2)
            plt.xlabel("Sigma")
            plt.ylabel("Mu")
            plt.show()
        else:
            return [Sigma[i], Mu[i], x1, x2]


# print(gestion_de_portefeuille_2actions(rendements_action1,rendements_action2))


# gestion de portefeuille avec 3 titres

def gestion_de_portefeuille_3actions(indice_t, methode, graphique=True):
    '''utilise la théorie de la gestion de portefeuille et donne la répartition des 3 titres la moins risquée à appliquer à l'instant indice+1'''
    if methode == 'AIC':
        pmax = 10
    else:
        pmax = 11
    port_3actions = ARCH_3titres(r1, r2, r3, indice_t, pmax, 15, 20, methode)
    inversible = port_3actions[6]
    if not inversible:
        return [0, 0, 0, 0, 0]
    else:
        Action1 = port_3actions[0]
        Action2 = port_3actions[1]
        Action3 = port_3actions[2]
        cov1_2 = covariance(port_3actions[3], port_3actions[4])
        cov1_3 = covariance(port_3actions[3], port_3actions[5])
        cov2_3 = covariance(port_3actions[4], port_3actions[5])
        step = 0.01
        L = []
        Mu1 = []
        Sigma1 = []
        for x1 in linspace(0, 1, 1 / step):
            for x2 in linspace(0, 1, 1 / step):
                if x1 + x2 < 1:
                    x3 = 1 - x1 - x2
                    mu = x1 * Action1[0] + x2 * Action2[0] + x3 * Action3[0]
                    sigma = (x1 ** 2) * Action1[1] + (x2 ** 2) * Action2[1] + (x3 ** 2) * Action3[
                        1] + 2 * x1 * x2 * cov1_2 + 2 * x1 * x3 * cov1_3 + 2 * x3 * x2 * cov2_3
                    L.append([sigma, mu, x1, x2, x3])
                    Mu1.append(mu)
                    Sigma1.append(sigma)

        Sigma = []
        for i in range(len(L)):
            Sigma.append(L[i][0])

        sigma_min = min(Sigma)

        i = 0
        while Sigma[i] != sigma_min:
            i += 1

        couple_optimal = [L[i][0], L[i][1]]  # coordonnées du point de risque minimum
        x1 = L[i][2]
        x2 = L[i][3]
        x3 = L[i][4]
        if graphique:
            print("le couple optimal est:")
            print("mu=", couple_optimal[1])
            print("sigma=", couple_optimal[0])
            print("La proportion optimale d action 1 est=", x1)
            print("La proportion optimale d action 2 est=", x2)
            print("La proportion optimale d action 3 est=", x3)

            plt.plot(Action1[1], Action1[0], "g+", markersize=17, mew=2)
            plt.plot(Action2[1], Action2[0], "g+", markersize=17, mew=2)
            plt.plot(Action3[1], Action3[0], "g+", markersize=17, mew=2)
            plt.plot(Sigma1, Mu1, "b+")
            plt.plot(couple_optimal[0], couple_optimal[1], 'r+', markersize=17, mew=2)
            plt.xlabel("Sigma")
            plt.ylabel("Mu")
            plt.show()
        else:
            return [Sigma1[i], Mu1[i], x1, x2, x3]


def gestion_de_portefeuille_4actions(indice_t, methode, graphique=True):
    '''utilise la théorie de la gestion de portefeuille et donne la répartition des 4 titres la moins risquée à appliquer à l'instant indice+1'''
    if methode == 'AIC':
        pmax = 10
    else:
        pmax = 11
    port_4actions = ARCH_4titres(r1, r2, r3, r4, indice_t, pmax, 15, 20, methode)
    inversible = port_4actions[8]
    if not inversible:
        return [0, 0, 0, 0, 0, 0]
    else:
        Action1 = port_4actions[0]
        Action2 = port_4actions[1]
        Action3 = port_4actions[2]
        Action4 = port_4actions[3]
        cov1_2 = covariance(port_4actions[4], port_4actions[5])
        cov1_3 = covariance(port_4actions[4], port_4actions[6])
        cov1_4 = covariance(port_4actions[4], port_4actions[7])
        cov2_3 = covariance(port_4actions[5], port_4actions[6])
        cov2_4 = covariance(port_4actions[5], port_4actions[7])
        cov3_4 = covariance(port_4actions[6], port_4actions[7])
        step = 0.01
        L = []
        Mu1 = []
        Sigma1 = []
        for x1 in linspace(0, 1, 1 / step):
            for x2 in linspace(0, 1, 1 / step):
                for x3 in linspace(0, 1, 1 / step):
                    if x1 + x2 + x3 < 1:
                        x4 = 1 - x1 - x2 - x3
                        mu = x1 * Action1[0] + x2 * Action2[0] + x3 * Action3[0] + x4 * Action4[0]
                        sigma = (x1 ** 2) * Action1[1] + (x2 ** 2) * Action2[1] + (x3 ** 2) * Action3[1] + (x4 ** 2) * \
                                Action4[
                                    1] + 2 * x1 * x2 * cov1_2 + 2 * x1 * x3 * cov1_3 + 2 * x1 * x4 * cov1_4 + 2 * x2 * x3 * cov2_3 + 2 * x2 * x4 * cov2_4 + 2 * x3 * x4 * cov3_4
                        L.append([sigma, mu, x1, x2, x3, x4])
                        Mu1.append(mu)
                        Sigma1.append(sigma)

        Sigma = []
        for i in range(len(L)):
            Sigma.append(L[i][0])

        sigma_min = min(Sigma)

        i = 0
        while Sigma[i] != sigma_min:
            i += 1

        couple_optimal = [L[i][0], L[i][1]]  # coordonnées du point de risque minimum
        x1 = L[i][2]
        x2 = L[i][3]
        x3 = L[i][4]
        x4 = L[i][5]
        if graphique:
            print("le couple optimal est:")
            print("mu=", couple_optimal[1])
            print("sigma=", couple_optimal[0])
            print("La proportion optimale d action 1 est=", x1)
            print("La proportion optimale d action 2 est=", x2)
            print("La proportion optimale d action 3 est=", x3)
            print("La proportion optimale d action 4 est=", x4)

            plt.plot(Action1[1], Action1[0], "g+", markersize=17, mew=2)
            plt.plot(Action2[1], Action2[0], "g+", markersize=17, mew=2)
            plt.plot(Action3[1], Action3[0], "g+", markersize=17, mew=2)
            plt.plot(Action4[1], Action4[0], "g+", markersize=17, mew=2)
            plt.plot(Sigma1, Mu1, "b+")
            plt.plot(couple_optimal[0], couple_optimal[1], 'r+', markersize=17, mew=2)
            plt.xlabel("Sigma")
            plt.ylabel("Mu")
            plt.show()
        else:
            return [Sigma1[i], Mu1[i], x1, x2, x3, x4]


def achat_2actions(P1, P2, indice_min, indice_max, cout_transac, capital, methode, graphique=True):
    '''méthode d'achat utilisant la gestion de portefeuille pour 2 titres sur un intervalle de temps donné et renvoyant le capital final possedé en actions'''
    l_capital = [capital]
    capital_initial = capital
    liquide = capital
    valeur_action = 0
    r_opt, risque_opt, x1, x2 = gestion_de_portefeuille_2actions(indice_min, methode, False)
    nb_action1 = x1 * capital // P1[indice_min]
    nb_action2 = x2 * capital // P2[indice_min]
    valeur_action = nb_action1 * P1[indice_min] + nb_action2 * P2[indice_min]
    liquide -= valeur_action * (1 + cout_transac)
    action1 = [nb_action1]
    action2 = [nb_action2]
    for i in range(indice_min + 1, indice_max + 1):
        valeur_action = nb_action1 * P1[i] + nb_action2 * P2[i]
        capital = liquide + valeur_action
        r_opt, risque_opt, x1, x2 = gestion_de_portefeuille_2actions(i, methode, False)
        if x1 != 0 or x2 != 0:
            nb_action1 = x1 * capital // P1[i]  # il faudrait prendre le nb_action le plus proche du capital
            nb_action2 = x2 * capital // P2[i]  # que l'on veut investir .... mais sans dépasser le capital
            action1.append(nb_action1)
            action2.append(nb_action2)
            diff_action1 = nb_action1 - action1[-2]
            diff_action2 = nb_action2 - action2[-2]
            valeur_action = nb_action1 * P1[i] + nb_action2 * P2[i]
            liquide -= (diff_action1 * P1[i] + diff_action2 * P2[i]) * (1 + cout_transac)
            if graphique:
                l_capital.append(liquide + valeur_action)
        else:
            if graphique:
                precedent = l_capital[-1]
                l_capital.append(precedent)
    if graphique:
        T_range = T1[indice_min:indice_min + len(l_capital)]
        T_graph = [t - int(T_range[0]) for t in T_range]
        plot(T_graph, l_capital)
        cap = int(capital * 10) / 10
        gain = int((capital / capital_initial - 1) * 10000) / 100
        plot(T_graph, [capital_initial] * len(T_graph),
             label="capital final : " + str(cap) + ", gain : " + str(gain) + "%")
        xlabel("Time (s)", fontsize=30)
        ylabel("Capital", fontsize=30)
        axes = gca()
        axes.xaxis.set_tick_params(labelsize=20)
        axes.yaxis.set_tick_params(labelsize=20)
        legend(prop={'size': 20})
        return T_graph, l_capital
    else:
        capital = liquide + valeur_action
    return capital, capital - capital_initial


def achat_3actions(P1, P2, P3, indice_min, indice_max, cout_transac, capital, methode, graphique=True):
    '''méthode d'achat utilisant la gestion de portefeuille pour 2 titres sur un intervalle de temps donné et renvoyant le capital final possedé en actions'''
    l_capital = [capital]
    capital_initial = capital
    liquide = capital
    valeur_action = 0
    r_opt, risque_opt, x1, x2, x3 = gestion_de_portefeuille_3actions(indice_min, methode, False)
    nb_action1 = x1 * capital // P1[indice_min]
    nb_action2 = x2 * capital // P2[indice_min]
    nb_action3 = x3 * capital // P3[indice_min]
    valeur_action = nb_action1 * P1[indice_min] + nb_action2 * P2[indice_min] + nb_action3 * P3[indice_min]
    liquide -= valeur_action * (1 + cout_transac)
    action1 = [nb_action1]
    action2 = [nb_action2]
    action3 = [nb_action3]
    for i in range(indice_min + 1, indice_max + 1):
        valeur_action = nb_action1 * P1[i] + nb_action2 * P2[i] + nb_action3 * P3[i]
        capital = liquide + valeur_action
        r_opt, risque_opt, x1, x2, x3 = gestion_de_portefeuille_3actions(i, methode, False)
        if x1 != 0 or x2 != 0 or x3 != 0:
            nb_action1 = x1 * capital // P1[i]  # il faudrait prendre le nb_action le plus proche du capital
            nb_action2 = x2 * capital // P2[i]  # que l'on veut investir .... mais sans dépasser le capital
            nb_action3 = x3 * capital // P3[i]
            action1.append(nb_action1)
            action2.append(nb_action2)
            action3.append(nb_action3)
            diff_action1 = nb_action1 - action1[-2]
            diff_action2 = nb_action2 - action2[-2]
            diff_action3 = nb_action3 - action3[-2]
            valeur_action = nb_action1 * P1[i] + nb_action2 * P2[i] + nb_action3 * P3[i]
            liquide -= (diff_action1 * P1[i] + diff_action2 * P2[i] + diff_action3 * P3[i]) * (1 + cout_transac)
            if graphique:
                l_capital.append(liquide + valeur_action)
        else:
            if graphique:
                precedent = l_capital[-1]
                l_capital.append(precedent)
    if graphique:
        T_range = T1[indice_min:indice_min + len(l_capital)]
        T_graph = [t - int(T_range[0]) for t in T_range]
        plot(T_graph, l_capital)
        cap = int(capital * 10) / 10
        gain = int((capital / capital_initial - 1) * 10000) / 100
        plot(T_graph, [capital_initial] * len(T_graph),
             label="capital final : " + str(cap) + ", gain : " + str(gain) + "%")
        xlabel("Time (s)", fontsize=30)
        ylabel("Capital", fontsize=30)
        axes = gca()
        axes.xaxis.set_tick_params(labelsize=20)
        axes.yaxis.set_tick_params(labelsize=20)
        legend(prop={'size': 20})
    else:
        capital = liquide + valeur_action
    return capital, capital - capital_initial


def achat_4actions(P1, P2, P3, P4, indice_min, indice_max, cout_transac, capital, methode, graphique=True):
    '''méthode d'achat utilisant la gestion de portefeuille pour 2 titres sur un intervalle de temps donné et renvoyant le capital final possedé en actions'''
    l_capital = [capital]
    capital_initial = capital
    liquide = capital
    valeur_action = 0
    r_opt, risque_opt, x1, x2, x3, x4 = gestion_de_portefeuille_4actions(indice_min, methode, False)
    nb_action1 = x1 * capital // P1[indice_min]
    nb_action2 = x2 * capital // P2[indice_min]
    nb_action3 = x3 * capital // P3[indice_min]
    nb_action4 = x4 * capital // P4[indice_min]
    valeur_action = nb_action1 * P1[indice_min] + nb_action2 * P2[indice_min] + nb_action3 * P3[
        indice_min] + nb_action4 * P4[indice_min]
    liquide -= valeur_action * (1 + cout_transac)
    action1 = [nb_action1]
    action2 = [nb_action2]
    action3 = [nb_action3]
    action4 = [nb_action4]
    for i in range(indice_min + 1, indice_max + 1):
        valeur_action = nb_action1 * P1[i] + nb_action2 * P2[i] + nb_action3 * P3[i] + nb_action4 * P4[i]
        capital = liquide + valeur_action
        r_opt, risque_opt, x1, x2, x3, x4 = gestion_de_portefeuille_4actions(i, methode, False)
        if x1 != 0 or x2 != 0 or x3 != 0 or x4 != 0:
            nb_action1 = x1 * capital // P1[i]  # il faudrait prendre le nb_action le plus proche du capital
            nb_action2 = x2 * capital // P2[i]  # que l'on veut investir .... mais sans dépasser le capital
            nb_action3 = x3 * capital // P3[i]
            nb_action4 = x4 * capital // P4[i]
            action1.append(nb_action1)
            action2.append(nb_action2)
            action3.append(nb_action3)
            action4.append(nb_action4)
            diff_action1 = nb_action1 - action1[-2]
            diff_action2 = nb_action2 - action2[-2]
            diff_action3 = nb_action3 - action3[-2]
            diff_action4 = nb_action4 - action4[-2]
            valeur_action = nb_action1 * P1[i] + nb_action2 * P2[i] + nb_action3 * P3[i] + nb_action4 * P4[i]
            liquide -= (diff_action1 * P1[i] + diff_action2 * P2[i] + diff_action3 * P3[i] + diff_action4 * P4[i]) * (
                        1 + cout_transac)
            if graphique:
                l_capital.append(liquide + valeur_action)
        else:
            if graphique:
                precedent = l_capital[-1]
                l_capital.append(precedent)
    if graphique:
        T_range = T1[indice_min:indice_min + len(l_capital)]
        T_graph = [t - int(T_range[0]) for t in T_range]
        plot(T_graph, l_capital)
        cap = int(capital * 10) / 10
        gain = int((capital / capital_initial - 1) * 10000) / 100
        plot(T_graph, [capital_initial] * len(T_graph),
             label="capital final : " + str(cap) + ", gain : " + str(gain) + "%")
        xlabel("Time (s)", fontsize=30)
        ylabel("Capital", fontsize=30)
        axes = gca()
        axes.xaxis.set_tick_params(labelsize=20)
        axes.yaxis.set_tick_params(labelsize=20)
        legend(prop={'size': 20})
    else:
        capital = liquide + valeur_action
    return capital, capital - capital_initial


# print(gestion_de_portefeuille_3actions(rendements_action1,rendements_action2,rendements_action3))

# gestion_de_portefeuille_2actions(1100,'AIC')
# gestion_de_portefeuille_3actions(6050)

# Sans discrétisation - 4h
# i_min = 12500
# i_max = 14000
# i_min = 100
# i_max = 20560
# i_min = 2822
# i_max = 2922
# i_min = 43125
# i_max = 48125
##Pour 1 minute - 4h
# i_min = 1000
# i_max = 1240

# cout_transac = 0.002
# capital = 10000
# methode = 'BIC'
# gain = achat_ARCH_1titre(r1,P1,i_min,i_max,capital,cout_transac,methode)
# print(gain)
# GAIN = achat_2actions(Pr1,Pr2,i_min,i_max,cout_transac,capital,methode)
# print(GAIN)
# t_graph,l_capital = achat_2actions(Pr1,Pr2,i_min,i_max,0,capital,methode)

# Gain = achat_3actions(Pr1,Pr2,Pr4,i_min,i_max,cout_transac,capital,methode)
# print(Gain)
# Gain = achat_3actions(Pr1,Pr2,Pr3,30,299,cout_transac,capital,methode)
# print(Gain)
# Gain = achat_4actions(Pr1,Pr2,Pr3,Pr4,i_min,i_max,cout_transac,capital,methode)
# print(Gain)
'''
imin = 1000
imax = 2940
--> 4h = 14400s
'''
'''
3 actions gain
i_min = 900
i_max = 1200
'''
'''
Credit agricole
Safran
debut credit agricole : 295962.67521
fin credit agricole : 297565.59239400004
'''

if __name__ == '__main__':
    P0 = 60
    directory_path = 'D:/HFT/stocks_secteurs/hft/Societe Generale.csv'
    df = pandas.read_csv(directory_path, header=0)
    r1 = list(df['log_return'])
    T1 = list(df['datetime'])

    directory_path = 'D:/HFT/stocks_secteurs/hft/Danone.csv'
    df = pandas.read_csv(directory_path, header=0)
    r2 = list(df['log_return'])
    T2 = list(df['datetime'])

    imin = 180
    imax = 1000

    p1 = P0;
    p2 = P0
    Pr1 = [];
    Pr2 = []
    for i in range(imax + 1):
        p1 = p1 * exp(r1[i]);
        p2 = p2 * exp(r2[i])
        Pr1.append(p1);
        Pr2.append(p2)

    cout_transac = 0.002
    capital = 10000
    methode = 'BIC'

    t_graph, l_capital = achat_2actions(Pr1, Pr2, imin, imax, 0, capital, methode)