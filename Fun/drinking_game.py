#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Sat Oct  9 10:55:38 2021@author: danielemamiriis"""import random# Dictionary der fortæller hvad de respektive konsekvenser til hvert terningeslag er. sum_udfald = {2 : "Vandfald!\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          3 : "Udnævn en træmand\n\n\n\n\n\n\n\n\n\n\n\n\n\n",           4 : "Giv 2 slurke væk\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          5 : "FÆLLES SKÅÅÅÅL\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          6 : "Kategori-leg. Taberen drikker 3 slurke\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          7 : "+1 toiletbesøg\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          8 : "Personen til højre drikker 2 slurke\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          9 : "Personen til venstre drikker 2 slurke\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          10 : "Tommelfinger på bordkanten – 3 slurke til taberen\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          11 : "Giv en udfordring – klares den, drikker du selv 4 slurke. Hvis ikke drikker den der tabte.\n\n\n\n\n\n\n\n\n\n\n\n\n\n",          12 : "Lav en regel / Fjern en regel\n\n\n\n\n\n\n\n\n\n\n\n\n\n"          }def slå_med_terningerne(resultat):    # Gemmer to terningekast og deres sum    terning_kast_1 = random.randint(1, 6)    terning_kast_2 = random.randint(1, 6)    sum = terning_kast_1 + terning_kast_2        # Printer de to terningekast    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nSlag:", terning_kast_1 , " og ", terning_kast_2)        # Hvis der bliver slået en tréer, drikker træmand    if terning_kast_1 == 3 and terning_kast_2 != 3:        print("Træmand drikker 1 slurk (er du træmand, så udnævn en ny)")            if terning_kast_2 == 3 and terning_kast_1 != 3:        print("Træmand drikker 1 slurk (er du træmand, så udnævn en ny)")                # Hvis der bliver slået 2 tréere, drikker træmand dobbelt    if terning_kast_1 == 3 and terning_kast_2 == 3:        print("Træmand drikker 2 slurke (er du træmand, så udnævn en ny)")        # Printer sum    print("Sum:", sum)        # Konsekvens af slag    return(sum_udfald[sum])#Skal bruges for at fortælle hvad brugeren skal gøre inden while-loopet går amokstart_spil = input("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nTryk på 'Enter' for at rulle med terningerne. Skriv 'stop' når spillet skal slutte: \n")# Itererer uendeligtwhile True:    print(slå_med_terningerne(sum_udfald))    if sum_udfald[3] == "Udnævn en træmand\n\n\n\n\n\n\n\n\n\n\n\n\n\n": # Ændrer udnævn        sum_udfald[3] = "Du drikker 2 slurke\n\n\n\n\n\n\n\n\n\n\n\n\n\n"    stop_spil = input() # Stopper automatisk while loopet, indtil der bliver trykket på enter    if stop_spil == "stop": # Medmindre der skrives stop, hvormed spillet slutter.        break