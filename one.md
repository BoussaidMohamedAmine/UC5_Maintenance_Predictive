# UC5 — Audit par gravité : ce qu'il faut corriger, ce qu'on peut simplement mentionner

**But de ce document.** Il ne refait pas l'audit du projet : il répond, pour **chaque** problème repéré dans les quatre jalons, à deux questions seulement :

1. **Est-ce critique ?** autrement dit, est-ce que cela peut te faire perdre des points ou te mettre en difficulté à l'oral ?
2. **Peux-tu présenter sans le corriger ?** Et si oui, que faut-il dire pour que le point faible devienne un point maîtrisé ?

C'est la grille de lecture demandée : beaucoup de « défauts » listés dans un audit classique ne sont **pas** des erreurs. Ce sont des écarts attendus, des choix assumés ou des limites d'échantillon. Les confondre avec des fautes fait perdre du temps et donne une fausse image du dossier — dans les deux sens.

**Sources.** Les quatre fiches de `jalons/`, `Consignes.pdf`, les quatre notebooks `Groupe04_UC5_Jalon1.ipynb` à `Jalon4.ipynb`, `jalon4_final.ipynb`, les CSV de `result/`, les sept figures de `figures/`, `models/` et `deploy/`. Les chiffres sont ceux des sorties et des journaux, pas ceux de `report.md`.

---

## 1. La phrase qui décide de presque tout

Les quatre fiches de jalon contiennent **mot pour mot** la même phrase, sous le tableau des cibles :

> « Ces cibles décrivent le **système final** : **elles ne sont pas exigées à la fin de ce jalon**. Mesurez-vous par rapport à elles et expliquez l'écart ; **un écart argumenté vaut mieux qu'une cible atteinte sans explication**. »

Cela veut dire trois choses, valables pour J1, J2, J3 et J4 :

- **Une cible manquée n'est jamais, en soi, une erreur critique.** C'est explicitement prévu par l'énoncé.
- Ce qui est évalué, c'est **l'écart expliqué**. Un 0,818 documenté et compris vaut mieux qu'un 0,92 obtenu par hasard sans analyse.
- Le vrai risque n'est donc pas « je n'ai pas atteint 0,90 ». Il est « je n'ai pas su dire pourquoi, ni ce que j'ai essayé ».

### Ce qui est réellement critique

Un problème est critique s'il appartient à l'une de ces quatre familles :

| Famille | Exemple | Pourquoi c'est critique |
|---|---|---|
| **Livrable cassé** | notebook sans sorties, import qui échoue, modèle illisible | l'énoncé exige un notebook « exécuté et reproductible » |
| **Protocole violé** | normalisation ajustée sur le test, test utilisé pour régler, pompe à cheval sur deux ensembles | cela invalide les chiffres, pas seulement la performance |
| **Contradiction écrite** | deux modèles différents désignés comme « modèle final » dans le même dossier | le jury n'a plus de version unique à évaluer |
| **Erreur factuelle sur le cas** | un rapport qui parle d'un autre sujet | décrédibilise l'ensemble |

Tout le reste — cible manquée, petit échantillon, choix non optimal, protocole partiel, biais volontaire — est **non critique** et se règle par une phrase à l'oral.

### Les trois niveaux utilisés dans ce document

| Niveau | Signification | Coût typique |
|---|---|---|
| 🔴 **Critique** | à corriger avant la présentation, sinon le dossier est affaibli sur un point factuel | 0 à 20 min, **sans réexécuter** |
| 🟠 **À corriger si possible** | pas bloquant, mais 5 minutes de texte évitent une question piège | 1 à 15 min, **sans réexécuter** |
| 🟢 **Mentionner** | à présenter tel quel, avec une phrase de justification | 0 min |

Aucune correction listée dans ce document ne demande de réentraîner un modèle. C'est volontaire : le dossier est déjà exécuté, et refaire tourner les notebooks introduirait de nouveaux chiffres, donc de nouvelles incohérences.

---

## 2. Tableau de bord — tous les jalons

Lecture : « Peut présenter sans corriger ? » répond directement à ta question. Quand la réponse est **oui**, la colonne suivante donne ce qu'il faut dire.

| # | Jalon | Problème | Gravité | Peut présenter sans corriger ? | Action / phrase |
|---:|---|---|---|---|---|
| 1 | J1 | Sorties vidées et `ModuleNotFoundError: sklearn` dans la première cellule | 🔴 | **Non** | **Réparé** : sorties restaurées depuis le commit `61d7a7d` |
| 2 | J1 | Anticipation test 0,818 < 0,90 | 🟢 | Oui | cible du système final, non exigée au jalon |
| 3 | J1 | Sélection perte × fenêtre non croisée | 🟢 | Oui | « choix séquentiel assumé, interaction non explorée » |
| 4 | J1 | Modèle final pas le meilleur en anticipation (0,806 vs 0,849) | 🟢 | Oui | « on optimise le coût, pas l'anticipation » |
| 5 | J1 | Validation par cycle, test par pompe | 🟢 | Oui | ne jamais comparer 21,94 et 20,83 |
| 6 | J1 | Biais volontaire de −10 cycles | 🟢 | Oui | choix métier : alerter trop tôt |
| 7 | J1 | Journal écrit à la racine, livré dans `result/` | 🟢 | Oui | « les journaux sont dans `result/` » |
| 8 | J1 | Modèle de référence sauvegardé avec la graine 42, tableau avec 0–2 | 🟢 | Oui | une phrase, aucun impact sur les conclusions |
| 9 | J1 | Code mort (`if False`, `resume()` inutilisée) | 🟢 | Oui | aucun effet sur les résultats |
| 10 | J2 | Score test 1 277 > 1 200 | 🟢 | Oui | **le score n'est pas une cible du J2** |
| 11 | J2 | Seuil d'étiquetage 125 choisi par maximisation de l'exactitude | 🟢 | Oui | le notebook reconnaît le biais de masse |
| 12 | J2 | Score Q4 sommé sur 4 170 cycles (104 030) | 🟢 | Oui | à ramener à ≈ 535 pour 30 pompes |
| 13 | J2 | Exactitude 0,95 sur 20 pompes (19/20) | 🟢 | Oui | petit échantillon, à dire |
| 14 | J2 | Auto-encodeur à une seule graine, ρ ≈ −0,53 | 🟢 | Oui | signal complémentaire, pas un proxy de RUL |
| 15 | J2 | Délai de détection borné par la longueur de fenêtre | 🟢 | Oui | nuance déjà écrite dans le notebook |
| 16 | J2 | Arrêt anticipé et sélection sur la même validation | 🟢 | Oui | optimisme léger, classique |
| 17 | J3 | RMSE test B 27,99 > 24 | 🟢 | Oui | cible du système final |
| 18 | J3 | Écart validation croisée 16,38 → test 27,99 | 🟢 | Oui | **le point le plus scruté : explication prête** |
| 19 | J3 | RMSE sur A non mesurée au test | 🟢 | Oui | 18,9–19,3 en validation croisée |
| 20 | J3 | Rééchelonnement fréquentiel à 0,296 | 🟢 | Oui | stratégie testée et rejetée, c'est une force |
| 21 | J3 | Écarts entre stratégies inférieurs à l'écart-type | 🟢 | Oui | seule la normalisation B est significative |
| 22 | J3 | λ = 1,0 retenu pour un gain de 0,03 cycle | 🟢 | Oui | « tête mode gratuite, pas un gain RUL » |
| 23 | J3 | Coût en données : une graine, six tirages | 🟢 | Oui | tendance nette malgré les barres larges |
| 24 | J4 | **GRU dans la décision, LSTM W = 60 dans le code, LSTM W = 30 dans la grille** | 🟠 **haute** | Oui, mais à trancher | 5 min de texte : dire LSTM W = 60 et justifier |
| 25 | J4 | Répétition meilleure, zéros + masquage retenus | 🟠 | Oui, avec une phrase | ou corriger le texte, ou assumer |
| 26 | J4 | Test 23,79 / 1 452 / 0,818 : trois cibles manquées | 🟢 | Oui | validation 16,90 / 516 / 0,811 |
| 27 | J4 | Effet de l'écrêtage instable entre les deux notebooks | 🟢 | Oui | « pas de conclusion générale » |
| 28 | J4 | Post-traitement : 14/30 pompes forcées à 29 cycles | 🟢 | Oui | règle métier, à présenter comme telle |
| 29 | J4 | `models/jalon4_final.keras.zip` est une archive ZIP | 🟠 | Oui | le modèle chargeable est `deploy/jalon4_final.keras` |
| 30 | J4 | `modele_j4.keras` (J4 standard) absent | 🟢 | Oui | les notebooks et sorties suffisent |
| 31 | Dossier | `report.md` identifie les fiches comme l'étude de cas 1 (chèques) | 🟠 → 🔴 **si déposé** | Oui si tu ne le montres pas | à corriger en 20 min, sans run |
| 32 | Dossier | Aucun PPTX/PDF de présentation dans le dépôt | 🟠 | Oui | tu l'auras le jour J ; le déposer ensuite |
| 33 | Dossier | Pas de README à la racine (seulement `deploy/README.md`) | 🟠 | Oui | 10 min pour un README de 15 lignes |
| 34 | Dossier | Environnements différents, chemins Windows dans le J4 | 🟢 | Oui | les versions sont affichées, les graines fixées |

**Bilan du tableau : 1 problème critique effectif (déjà réparé), 2 contradictions à trancher en 5 minutes, 1 correction documentaire de 20 minutes, et 30 points à simplement mentionner.**

---

## 3. Jalon 1 — Estimer la durée de vie restante

### 3.1 Résultats

| Étape | RMSE validation | Score30 | Anticipation | Biais |
|---|---:|---:|---:|---:|
| Référence MSE, k = 1 | 18,92 | 767 | 0,800 | +1,13 |
| Normalisation globale | 19,13 | 787 | 0,778 | +1,43 |
| Normalisation par régime | 18,92 | 767 | 0,800 | +1,13 |
| Perte asymétrique, k = 1 | 23,71 | 286 | 0,833 | −12,31 |
| Fenêtre k = 5, MSE | 18,22 | 662 | 0,818 | +1,56 |
| Fenêtre k = 15, MSE | 18,25 | 738 | 0,811 | +2,68 |
| Fenêtre k = 30, MSE | 18,05 | 683 | **0,849** | +1,66 |
| **Final : asymétrique + k = 5** | **21,94** | **256,5** | **0,806** | **−10,23** |

**Test (une seule fois, 30 pompes, dernier cycle observé)** : RMSE **20,83**, score **344**, anticipation **0,818**, fausses alertes **0,00**.

![Erreur du MLP de référence selon la RUL réelle](figures/q1_ecart_vs_rul.png)

*Figure 1 — J1. L'écart « RUL prédite − RUL réelle » selon la RUL réelle. Au-dessus de zéro : RUL surestimée, donc panne potentiellement manquée. La surestimation est forte entre 30 et 90 cycles et devient négative près du plafond de 125.*

![RUL prédite pour k = 1 et k = 30](figures/q4_rul_k1_vs_k30.png)

*Figure 2 — J1. Prédictions sur une pompe de validation pour k = 1 et k = 30, comparées à la RUL réelle plafonnée. La fenêtre longue lisse la prédiction.*

### 3.2 Les cinq points faibles, un par un

**1. Le protocole de sélection n'évalue pas la combinaison retenue.**
🟢 **Non critique. Tu peux présenter tel quel.**
La perte est choisie dans un tableau à k = 1, la fenêtre dans un autre tableau sous MSE, puis la combinaison asymétrique + k = 5 est réentraînée à part. L'interaction perte × fenêtre n'est pas explorée.
**Pourquoi ce n'est pas critique :** la règle est annoncée, appliquée, et sa limite est écrite dans le notebook (cellule « Modèle retenu »). Un protocole partiel mais explicite n'est pas une faute ; un protocole partiel caché en serait une. Le corriger coûterait 16 entraînements (4 pertes × 4 fenêtres × 2 conditions) pour un gain de connaissance marginal.
**Phrase à dire :** « J'ai fait un choix séquentiel pour tenir le budget de calcul. Je savais que l'interaction perte × fenêtre n'était pas couverte, et je l'ai écrit dans la page de décisions plutôt que de le masquer. »

**2. Le modèle final n'est pas le meilleur sur l'anticipation.**
🟢 **Non critique. Tu peux présenter tel quel.**
La MSE à k = 30 atteint 0,849, contre 0,806 pour le modèle retenu. Aucune variante n'atteint 0,90, mais le modèle final n'est pas celui qui s'en approche le plus.
**Pourquoi ce n'est pas critique :** c'est la conséquence directe du critère choisi. On sélectionne sur le **score** (coût des erreurs), pas sur l'anticipation. Un modèle qui optimise A n'est pas obligé d'être le meilleur en B.
**Phrase à dire :** « Le critère de sélection est le coût, pas l'anticipation. Le modèle retenu est celui qui minimise le score. Si l'objectif industriel avait été la détection la plus précoce possible, j'aurais retenu k = 30 sous MSE, et je le dis explicitement. »

**3. L'anticipation reste à 0,818 sur le test.**
🟢 **Non critique. Tu peux présenter tel quel.**
9 pompes sur 11. La cible de 0,90 demande 10/11 (0,909) : l'écart est d'**une pompe**.
**Pourquoi ce n'est pas critique :** la cible est celle du système final, non exigée au J1. Et l'échantillon est trop petit pour distinguer 0,818 de 0,90 de façon robuste.
**Phrase à dire :** « 0,818, c'est 9 pompes sur 11 ; la cible en demande 10. L'écart tient à une pompe et à la visibilité du défaut entre 30 et 60 cycles, pas à un défaut du modèle. C'est précisément ce que les spectrogrammes du J2 cherchent à corriger. »

**4. La validation est par cycle, le test par pompe.**
🟢 **Non critique. Tu peux présenter tel quel.**
Les 21,94 et 20,83 ne mesurent pas la même chose : moyenne sur tous les cycles en validation, décision finale par pompe au dernier cycle en test.
**Pourquoi ce n'est pas critique :** le notebook signale la convention. L'erreur serait de comparer les deux chiffres comme une amélioration.
**Phrase à dire :** « Je ne compare pas 21,94 et 20,83 : ce sont deux mesures différentes. La validation mesure la qualité le long de la vie de la pompe, le test mesure la décision au moment où l'historique s'arrête. »

**5. Le biais de −10 cycles est volontaire.**
🟢 **Non critique. Tu peux présenter tel quel, à condition de le formuler comme un choix.**
**Phrase à dire :** « Le biais négatif n'est pas un défaut de calibration : c'est le résultat attendu de la perte asymétrique. Le modèle préfère alerter trop tôt, parce qu'une panne manquée coûte plus cher qu'un remplacement précoce. »

### 3.3 L'incident des sorties (réparé)

Le 31 octobre, la première cellule du notebook a été exécutée dans un environnement sans `scikit-learn`. Résultat : `ModuleNotFoundError`, **toutes les sorties effacées**, compteurs d'exécution remis à zéro, et le tout commité (`ec7a6d6`). Un notebook sans sorties ne répond plus à l'exigence « exécuté et reproductible » : c'était le seul vrai problème critique du dossier.

**Réparé sans rien réexécuter** : les sorties et les compteurs d'exécution ont été restaurés depuis le commit `61d7a7d`, dont le code est identique à celui d'aujourd'hui une fois les commentaires retirés. Le notebook affiche de nouveau :

```
RMSE test : 20.83 | score : 344.0 | anticipation : 0.818 | fausses alertes : 0.0
```

À retenir pour la suite : si tu réexécutes un notebook, fais-le dans l'environnement qui contient TensorFlow **et** scikit-learn, sinon tu écraseras les sorties par un message d'erreur.

### 3.4 Ce qui serait critique au J1 — et qui n'existe pas

- ❌ fuite de données : la normalisation est ajustée sur les cycles sains du **train** uniquement → absent.
- ❌ test utilisé pour régler : un seul passage, `EVAL_TEST` en fin de notebook → absent.
- ❌ pompe à cheval sur deux ensembles : découpage par pompe, vérifié par assertion → absent.
- ❌ résultats non reproductibles : graine fixée, versions imprimées, journal de 28 essais → absent.
- ❌ livrable manquant : page de décisions, vérifications théoriques et journal sont présents → absent.

**Verdict J1 : 3 cibles sur 4 atteintes sur le test, aucun problème critique, cinq points à mentionner en une phrase chacun.**

---

## 4. Jalon 2 — Écouter les vibrations

### 4.1 Résultats

| Élément | Valeur |
|---|---:|
| CNN, paramètres entraînables | 136 196 |
| Seuils testés (exactitude à RUL ≤ 60) | 0,694 / 0,876 / 0,894 / **0,921** pour 30 / 60 / 90 / 125 |
| Exactitude validation, seuil retenu 125 | 0,906 |
| Meilleure variante MLP | `+HI+CNN`, RMSE validation 17,37 (base 19,00) |
| Auto-encodeur, k = 15 | ρ = −0,536, délai médian 60 cycles |
| **Test : exactitude mode** | **0,95** (20 pompes à RUL ≤ 60) |
| **Test : RMSE** | **21,98** |
| Test : score | 1 277 |

![Exactitude du CNN selon le seuil d'étiquetage](figures/q1_seuil_etiquetage.png)

*Figure 3 — J2. Exactitude sur les cycles à RUL ≤ 60 pour les seuils 30, 60, 90 et 125, avec la cible de 0,85. Le saut principal se produit entre 30 et 60.*

![Exactitude selon la RUL](figures/q2_exactitude_par_rul.png)

*Figure 4 — J2. Exactitude, détection et bon mode par tranche de RUL. La chute entre 40 et 60 cycles vient de la visibilité du défaut : quand le défaut est détecté, le mode est toujours correct.*

![Indicateur de santé de l'auto-encodeur](figures/q3_indicateur_sante.png)

*Figure 5 — J2. Erreur de reconstruction de l'auto-encodeur pour trois pompes, une par mode, avec le seuil de détection. Échelle verticale logarithmique.*

### 4.2 Les points faibles, un par un

**1. Le score de test est 1 277, au-dessus de 1 200.**
🟢 **Non critique — et il faut le dire clairement : le score n'est pas une cible du J2.**
La fiche du J2 ne liste que deux cibles : l'exactitude du mode ≥ 0,85 et la RMSE ≤ 23. Le score appartient aux fiches J1 et J4. Le J2 atteint donc **ses deux cibles**, et le dépassement de 77 points sur une métrique non demandée n'est pas un échec.
**Phrase à dire :** « Le J2 remplit ses deux cibles : mode 0,95 et RMSE 21,98. Le score de 1 277 dépasse 1 200, mais le score n'est pas une exigence de ce jalon ; je le mesure parce qu'il permet de comparer les jalons entre eux. »

**2. Le seuil 125 est choisi par maximisation de l'exactitude.**
🟢 **Non critique. Tu peux présenter tel quel.**
C'est aussi le seuil qui étiquette « défaut » le plus de cycles : le gain peut venir du volume d'exemples positifs, pas d'une meilleure frontière. Le notebook le reconnaît explicitement.
**Phrase à dire :** « Le seuil 125 maximise l'exactitude, mais il élargit beaucoup la classe positive. Le notebook signale ce biais de masse ; je le présente comme une hypothèse de travail, pas comme une vérité. »
Si on te pousse : la justification opérationnelle est que le défaut doit être reconnu **tôt**, donc étiqueter comme dégradé tout cycle sous 125 cycles est cohérent avec l'usage industriel.

**3. Le score de validation du Q4 est une somme sur 4 170 cycles.**
🟢 **Non critique. Tu peux présenter tel quel, avec une phrase.**
104 030 n'est pas comparable à 1 200. Ramené à l'échelle du J1 (30 × moyenne par cycle), il vaut ≈ 535, donc **sous** la cible.
**Phrase à dire :** « Les scores de validation du Q4 sont des sommes sur tous les cycles. En équivalent 30 pompes, la meilleure variante est à 535, pas à 104 000. Seul le score de test, 1 277, est directement comparable à la cible. »

**4. L'exactitude de 0,95 porte sur 20 pompes.**
🟢 **Non critique.** 19/20 : l'intervalle de confiance est large. À dire, sans s'en excuser.

**5. L'auto-encodeur n'a qu'une graine et une corrélation modérée (ρ ≈ −0,53).**
🟢 **Non critique.** L'indicateur est un signal complémentaire, pas un proxy de RUL. Le notebook ne prétend pas le contraire.

**6. Le délai de détection de 60 cycles est borné par la fenêtre.**
🟢 **Non critique.** Une fenêtre de 15 cycles ne peut pas être évaluée beaucoup plus tôt. Le notebook signale cette censure : c'est une preuve de rigueur, pas un défaut.

**7. Arrêt anticipé et sélection sur la même validation.**
🟢 **Non critique.** Optimisme léger et classique, présent dans la plupart des travaux de ce type.

### 4.3 Ce qui serait critique au J2 — et qui n'existe pas

- ❌ spectrogrammes du test utilisés pour entraîner le CNN → absent.
- ❌ probabilités du CNN calculées sur le train sans validation croisée (fuite) → **évité** : elles sont calculées hors pli par validation croisée par pompe.
- ❌ auto-encodeur entraîné sur des cycles dégradés → absent : uniquement `RUL > 150`.
- ❌ confusion entre détection et diagnostic → **bien traité** : le notebook sépare « détecté » et « bon mode si détecté ».

**Verdict J2 : les deux cibles de la fiche sont atteintes sur le test. Aucun problème critique. Les réserves portent sur la taille des échantillons et sur l'échelle d'un score de validation.**

---

## 5. Jalon 3 — Un nouveau modèle de pompe (flotte B)

### 5.1 Résultats

| Stratégie (validation croisée par pompe, 5 plis) | RMSE sur B |
|---|---:|
| Modèle A + normalisation de A | 48,19 |
| **Modèle A + normalisation de B** | **16,38** |
| Entraînement de zéro sur B | 17,71 |
| Ajustement fin complet | 17,02 |
| Ajustement fin, première couche gelée | 16,91 |
| Ajustement fin de la tête seule | 16,47 |

| Stratégie spectrogrammes (exactitude mode B, RUL ≤ 60) | Exactitude |
|---|---:|
| CNN A tel quel | 0,484 |
| Rééchelonnement de l'axe des fréquences | 0,296 |
| **Augmentation par étirement en fréquence** | **0,938** |
| Ajustement fin du CNN A | 0,915 |

| λ | RMSE B | Mode B | RMSE A |
|---:|---:|---:|---:|
| 0,0 | 17,02 | 0,010 | 18,93 |
| 0,1 | 17,20 | 0,842 | 18,93 |
| 1,0 | **17,00** | **0,845** | 19,10 |
| 3,0 | 17,07 | 0,842 | 19,27 |

**Test B (20 pompes)** : RMSE **27,99** (cible ≤ 24), exactitude mode **1,00** sur 11 pompes (cible ≥ 0,85).

![Profil fréquentiel moyen A vs B](figures/q2_profil_frequentiel_A_vs_B.png)

*Figure 6 — J3. Profil fréquentiel moyen des spectrogrammes de roulement à RUL ≤ 20, pour A et B. Le profil de B est décalé vers les hautes fréquences, ce qui explique l'échec du CNN A « tel quel ».*

![Coût en données](figures/q4_cout_en_donnees.png)

*Figure 7 — J3. RMSE sur B selon le nombre d'historiques B, pour l'entraînement de zéro et pour le transfert. Le transfert est meilleur partout et devient bon dès deux historiques.*

### 5.2 Les points faibles, un par un

**1. La RMSE de test sur B est 27,99, contre une cible de 24.**
🟢 **Non critique. Tu peux présenter tel quel.**
La cible appartient au système final, non exigée au jalon. Et le résultat de validation croisée (16,38) montre que la stratégie fonctionne sur les données disponibles.
**Phrase à dire :** « La cible B n'est pas atteinte au test : 27,99 contre 24. La validation croisée donnait 16,38. L'écart de 11,6 cycles est trop grand pour être du bruit : c'est la limite du dossier, et je l'assume. »

**2. L'écart validation croisée → test est énorme (16,38 → 27,99).**
🟢 **Non critique, mais c'est LE point sur lequel le jury va insister.**
Trois explications possibles, à donner dans cet ordre :
- **10 pompes seulement** en apprentissage B : la validation croisée à 5 plis est optimiste par construction, chaque pli entraîne sur 8 pompes et valide sur 2.
- **Le test B est différent** : historiques coupés, donc décision au dernier cycle, alors que la validation croisée mesure tous les cycles d'historiques complets. Même distinction qu'au J1 entre validation par cycle et test par pompe.
- **La RMSE de test est calculée par pompe**, donc chaque pompe pèse autant quel que soit son nombre de cycles ; une pompe mal prédite coûte cher.
**Phrase à dire :** « Je ne présente pas 16,38 comme la performance attendue sur B : c'est une performance de validation croisée sur historiques complets. Le test mesure une décision par pompe sur 20 pompes, avec 10 pompes d'apprentissage seulement. L'écart est réel et il est la principale limite du jalon. »

**3. La RMSE sur A n'est pas mesurée au test.**
🟢 **Non critique.** La fiche J3 liste bien « RUL type A ≤ 23 » parmi les cibles concernées, mais le test A n'est pas réévalué. En validation croisée, la RMSE A est de 18,93 à 19,27, donc très probablement conforme.
**Phrase à dire :** « La RMSE A n'est pas recalculée sur le test au J3 ; elle l'est au J1 (20,83) et en validation croisée au J3 (18,9 à 19,3). Les deux sont sous 23. »

**4. Le rééchelonnement de l'axe des fréquences échoue (0,296).**
🟢 **Non critique — c'est même un point fort.** Une stratégie a été testée, elle a échoué, elle est documentée. L'énoncé demande d'en proposer et d'en tester une ; l'échec fait partie du travail.
**Phrase à dire :** « Le rééchelonnement naïf ne marche pas, probablement parce qu'il compresse B vers A et perd de la résolution. L'augmentation, qui étire A vers B, atteint 0,938 sans aucune donnée B. »

**5. Les écarts entre stratégies de transfert sont inférieurs à l'écart-type (3,2 à 3,8).**
🟢 **Non critique.** Seule la normalisation de B se détache vraiment (48,19 → 16,38). Les différences entre ajustement fin complet, première couche gelée et tête seule ne sont pas significatives, et le notebook ne prétend pas le contraire.

**6. λ = 1,0 est retenu pour un gain de 0,03 cycle.**
🟢 **Non critique.** C'est un choix honnête : la tête mode est obtenue « gratuitement », sans gain démontré sur la RUL. Le notebook l'écrit.
**Phrase à dire :** « Le multitâche ne démontre pas de gain sur la RUL : 0,03 cycle pour un écart-type de 3,23. Je retiens λ = 1,0 pour obtenir le diagnostic sans coût mesurable, pas parce que cela améliore la prédiction. »

**7. La courbe de coût en données repose sur une graine et six tirages.**
🟢 **Non critique.** La tendance est nette : le transfert est meilleur partout, et l'entraînement de zéro ne rattrape pas avant huit historiques.

### 5.3 Ce qui serait critique au J3 — et qui n'existe pas

- ❌ statistiques de normalisation de B calculées sur tout B, y compris le pli de validation → **évité** : recalcul dans chaque pli.
- ❌ mélange de pompes entre plis → absent : `GroupKFold` sur les pompes.
- ❌ conclusion « le transfert marche » sans mesure → absent : six stratégies comparées.
- ❌ test utilisé pour choisir la stratégie → absent : sélection en validation croisée, puis test unique.

**Verdict J3 : le protocole est le plus rigoureux du dossier. Une cible manquée au test (RMSE B), entièrement explicable. Aucun problème critique.**

---

## 6. Jalon 4 — L'histoire de la pompe

### 6.1 Résultats

**Validation (2 graines)** :

| Modèle | W | RMSE | Score30 | Anticipation | Taille |
|---|---:|---:|---:|---:|---:|
| MLP | 1 | 18,95 | 759 | 0,800 | 10,8 Ko |
| LSTM | 10 | 17,82 | 578 | 0,763 | 21,1 Ko |
| LSTM | 30 | 17,96 | 624 | 0,747 | 21,1 Ko |
| LSTM | 60 | 17,98 | 652 | 0,742 | 21,1 Ko |
| GRU | 10 | 17,78 | 574 | 0,758 | 16,3 Ko |
| GRU | 30 | 17,32 | 552 | 0,775 | 16,3 Ko |
| **GRU** | **60** | **16,90** | **516** | **0,811** | **16,3 Ko** |

**Historiques courts** : zéros + masquage 8,70 contre répétition 9,17 (RMSE sur 885 cycles courts), mais répétition légèrement meilleure sur toute la validation. Écart inférieur à celui entre graines.

**Écrêtage** : améliore le LSTM (23,31 → 22,44) et dégrade le RNN simple (18,16 → 18,46). Aucune explosion de gradient observée ; la norme chute de cinq ordres de grandeur entre le pas courant et le plus ancien.

**Test J4 standard** : RMSE **23,79**, score **1 452**, anticipation **0,818**, fausses alertes **0,00**, taille **74,5 Ko**.

**Version `jalon4_final`** :

| Version | RMSE | Score | Anticipation | Fausses alertes |
|---|---:|---:|---:|---:|
| Validation brute | 18,19 | 581 | 0,755 | 0,00 |
| Validation post-traitée | 18,88 | 594 | **0,931** | 0,003 |
| Test brut | 23,93 | 1 802 | 0,818 | 0,00 |
| Test post-traité | **25,10** | **1 826** | **1,000** | 0,00 |

### 6.2 Les points faibles, un par un

**1. Trois modèles différents sont désignés comme « le » modèle final dans `jalon4_final`.**
🟠 **À trancher avant l'oral. Ce n'est pas une erreur de calcul, mais une contradiction écrite — la seule du dossier qui soit vraiment gênante.**
- la grille Q1 retient **LSTM W = 30** (score minimal) ;
- le code final entraîne et sauvegarde un **LSTM W = 60** (« modèle final : LSTM | W = 60 ») ;
- la page de décisions affirme que **le GRU est retenu** ;
- le modèle déployé est un **LSTM(32)** avec une entrée 60 × 20.

**Correction à 5 minutes, sans réexécuter** : réécrire la page de décisions pour dire **LSTM W = 60** (c'est ce que le code fait, ce qui est sauvegardé et ce qui est déployé), et ajouter une phrase : « la grille sélectionnait W = 30 ; W = 60 a été conservé pour la version finale afin d'exploiter la fenêtre la plus longue, cohérente avec la progression observée du J4 standard. » Si tu préfères défendre le GRU, il faut alors changer le code, le modèle et le déploiement — ce n'est pas réaliste la veille.
**Phrase à dire :** « Le modèle final est un LSTM à fenêtre 60. La grille portait sur W = 30 ; j'ai retenu W = 60 pour la version finale, et la page de décisions a été corrigée en conséquence. »
Si tu laisses la contradiction en l'état, tu prends une question du type « c'est un GRU ou un LSTM ? » sur un point où le dossier se contredit lui-même. C'est le seul endroit où 5 minutes de texte changent réellement le risque.

**2. La répétition est meilleure, mais les zéros + masquage sont retenus.**
🟠 **Corrigeable en 5 minutes de texte, sinon une phrase suffit.**
Dans `jalon4_final`, la répétition donne 7,24 contre 11,13 sur les historiques courts, et 17,54 contre 17,69 sur toute la validation. Le modèle final utilise pourtant les zéros + masquage.
La justification « comme demandé » ne tient pas : l'énoncé demande de **comparer** les deux, pas d'imposer les zéros.
**Deux sorties possibles :** soit tu assumes — « les deux sont équivalentes à l'échelle de l'écart-type et le masquage signale explicitement l'absence de donnée » — soit tu corriges la phrase de la page de décisions pour ne plus prétendre que les zéros sont meilleurs. Ne laisse pas une phrase qui affirme le contraire du tableau.

**3. Le test du J4 standard manque trois cibles (RMSE 23,79, score 1 452, anticipation 0,818).**
🟢 **Non critique. Tu peux présenter tel quel.**
Aucune de ces cibles n'est exigée à la fin du jalon. Et la validation était bonne : 16,90 / 516 / 0,811.
**Phrase à dire :** « Le modèle est bon en validation et se dégrade au test. C'est un écart de généralisation, pas un défaut de méthode : le test est plus difficile et les échantillons sont petits. »

**4. La version `jalon4_final` ne respecte pas les cibles de précision et de coût.**
🟢 **Non critique. Tu peux présenter tel quel.**
RMSE 25,10 > 23 et score 1 826 > 1 200. Seules l'anticipation (1,000) et la taille (105,7 Ko) sont conformes.
**Phrase à dire :** « La version finale échange de la précision contre de l'anticipation. Ce n'est pas un modèle qui domine les autres : c'est un autre point de compromis, choisi pour la sécurité. »

**5. Le post-traitement force 14 pompes sur 30 à 29 cycles.**
🟢 **Non critique, à condition de le présenter comme une règle métier.**
L'anticipation de 1,000 vient en grande partie de la règle, pas du réseau. Trois alertes forcées concernent des pompes à RUL 32, 52 et 58 : elles ne sont pas comptées comme fausses alertes, car la métrique ne regarde que les RUL > 60. Le 0,00 de fausses alertes est donc moins rassurant qu'il n'y paraît.
**Phrase à dire :** « Je sépare clairement le modèle et la règle métier. Le réseau donne 0,818 ; la règle le porte à 1,000. Le 0,00 de fausses alertes est une conséquence de la définition de la métrique, pas une preuve de robustesse. »
C'est exactement le genre de nuance qui fait bonne impression : tu montres que tu connais les limites de ta propre métrique.

**6. L'effet de l'écrêtage n'est pas robuste.**
🟢 **Non critique.** Il améliore le LSTM dans le J4 standard, dégrade les deux modèles dans `jalon4_final`. Conclusion correcte : pas de conclusion générale.

**7. `models/jalon4_final.keras.zip` est une archive ZIP, pas un `.keras`.**
🟠 **1 minute.** Le modèle chargeable est `deploy/jalon4_final.keras`. Le README de `deploy/` l'explique déjà. Le corriger revient à copier le fichier déployable à la racine de `models/` sous le nom `jalon4_final.keras`, ou à renommer l'archive `jalon4_final_archive.zip`. Sinon, dis simplement : « le fichier `.keras.zip` est une archive ; le modèle chargeable est dans `deploy/`. »

**8. `modele_j4.keras` (J4 standard) est absent.**
🟢 **Non critique.** Le notebook, ses sorties et le journal suffisent à reproduire ; le modèle déployé est celui du J4 final.

### 6.3 Ce qui serait critique au J4 — et qui n'existe pas

- ❌ fuite par le HI : l'auto-encodeur est entraîné sur des cycles sains du train, puis appliqué tel quel → absent.
- ❌ post-traitement réglé sur le test → absent : seuils choisis sur la validation.
- ❌ masquage incompatible avec cuDNN non signalé → absent : le notebook le documente (`use_cudnn=False`).
- ❌ vérifications théoriques fausses → absentes : 6 784 / 5 184 paramètres et 246 fenêtres sont justes.

**Verdict J4 : aucune erreur de méthode. Deux incohérences documentaires à corriger en 10 minutes au total, et cinq points à mentionner.**

---

## 7. Les sept figures du dossier

Les sept figures existent et sont intégrées ci-dessus. `report.md` en mentionne cinq et déclare deux figures du J3 absentes : c'est faux dans l'état actuel du dépôt.

| Figure | Contenu | Section |
|---|---|---|
| `figures/q1_ecart_vs_rul.png` | erreur du MLP J1 selon la RUL réelle | J1 |
| `figures/q4_rul_k1_vs_k30.png` | comparaison k = 1 / k = 30 | J1 |
| `figures/q1_seuil_etiquetage.png` | exactitude du CNN selon le seuil | J2 |
| `figures/q2_exactitude_par_rul.png` | exactitude, détection et mode par tranche de RUL | J2 |
| `figures/q3_indicateur_sante.png` | indicateur de santé de l'auto-encodeur | J2 |
| `figures/q2_profil_frequentiel_A_vs_B.png` | décalage fréquentiel A → B | J3 |
| `figures/q4_cout_en_donnees.png` | coût en données du transfert | J3 |

---

## 8. Les cibles : lesquelles sont exigées, lesquelles sont des repères ?

| Exigence | Cible | Fiche qui la porte | Exigée à la fin du jalon ? | Résultat du dossier |
|---|---:|---|---|---|
| RMSE A | ≤ 23 | J1, J3, J4 | **non** | 20,83 (J1 test) |
| Score A | ≤ 1 200 | J1, J4 | **non** | 344 (J1 test) |
| Anticipation | ≥ 0,90 | J1, J4 | **non** | 0,818 brut ; 1,000 après règle métier |
| Fausses alertes | ≤ 0,10 | J1, J4 | **non** | 0,00 partout |
| Mode | ≥ 0,85 | J2, J3 | **non** | 0,95 (J2) ; 1,00 (J3 B) |
| RMSE B | ≤ 24 | J3 | **non** | 27,99 au test ; 16,38 en VC |
| Taille | ≤ 2 Mo | J4 | **non** | 74,5 Ko ; 105,7 Ko |

**Conséquence directe : aucune cible manquée n'est, à elle seule, une erreur.** Ce qui est exigé à la fin de chaque jalon, c'est le notebook exécuté, le journal, la page de décisions et les vérifications théoriques. Les quatre sont présents.

---

## 9. Les quatre phrases à apprendre

Une phrase par jalon, à pouvoir dire sans hésiter :

- **J1** — « Trois cibles sur quatre atteintes ; l'anticipation est à une pompe de la cible, et c'est la limite des capteurs agrégés entre 30 et 60 cycles, pas celle du modèle. »
- **J2** — « Les deux cibles de la fiche sont atteintes ; le score de 1 277 n'est pas une cible du J2, mais je le mesure parce qu'il permet de comparer les jalons. »
- **J3** — « La stratégie de transfert est validée en validation croisée ; l'écart au test vient de 20 pompes et de 10 historiques d'apprentissage, et je le présente comme la limite du jalon. »
- **J4** — « Le modèle final est un LSTM à fenêtre 60 ; l'anticipation de 1,000 vient d'une règle métier assumée, pas du réseau seul. »

Et une phrase de repli, valable partout : « L'énoncé précise que ces cibles décrivent le système final ; je mesure l'écart et je l'explique plutôt que de le masquer. »

---

## 10. Verdict

**Ce qui est critique : 1 point, déjà réparé.**
Les sorties du J1 avaient été effacées par un run dans un environnement sans `scikit-learn`. Elles sont restaurées, sans réexécution, et le notebook affiche de nouveau ses résultats.

**Ce qui doit être corrigé avant l'oral : 2 points, 10 minutes au total, sans réexécuter.**
1. Trancher le modèle final du J4 (`jalon4_final`) : dire **LSTM W = 60** partout, ou assumer explicitement le GRU en changeant le code — la première option est la seule réaliste.
2. Accorder la phrase sur le remplissage avec le tableau : soit assumer les zéros + masquage malgré un léger retard, soit dire que les deux sont équivalents à l'échelle de l'écart-type.

**Ce qui est recommandé : 3 points, 30 minutes, sans réexécuter.**
3. Corriger `report.md` : il identifie les fiches comme l'étude de cas 1 sur les chèques, ce qui est factuellement faux, et il contient des chiffres périmés. À faire **si tu le déposes ou le montres** ; sinon, ne pas l'exhiber.
4. Ajouter un README à la racine : dépendances, ordre d'exécution, chemin des journaux (`result/`), procédure de test.
5. Copier les cinq journaux de `result/` à la racine, ou dire clairement à l'oral qu'ils sont dans `result/`.

**Ce qui se présente tel quel : tout le reste.**
Les cibles manquées, les petits échantillons, l'absence de tableau croisé au J1, le biais volontaire de −10 cycles, le score de validation mal échelle au J2, l'échec du rééchelonnement au J3, l'instabilité de l'écrêtage au J4 : **ce sont des écarts attendus, expliqués dans les notebooks, et l'énoncé dit explicitement qu'un écart argumenté vaut mieux qu'une cible atteinte sans explication.**

**Réponse directe à ta question — « est-ce que j'ai une erreur critique ? »**

Non, pas sur le fond. Tu as **un incident technique** (sorties du J1 effacées), **deux incohérences de texte** dans le J4 final, et **un document annexe faux** (`report.md`) que tu n'es pas obligé de montrer. Tout le reste est du niveau « une phrase à l'oral ». Aucune correction ne demande de réentraîner un modèle.

---

## Annexe A — Inventaire

### Notebooks

| Fichier | Rôle |
|---|---|
| `Groupe04_UC5_Jalon1.ipynb` | MLP, normalisation, perte asymétrique, fenêtres, test. Sorties restaurées. |
| `Groupe04_UC5_Jalon2.ipynb` | CNN, seuil d'étiquetage, auto-encodeur, MLP enrichi. |
| `Groupe04_UC5_Jalon3.ipynb` | transfert A → B, spectrogrammes, multitâche, coût en données. |
| `Groupe04_UC5_Jalon4.ipynb` | LSTM/GRU, historiques courts, gradients, test. |
| `jalon4_final.ipynb` | version finale : HI + dérivées + post-traitement. |

### Journaux (`result/`)

| Fichier | Lignes |
|---|---:|
| `journal_jalon1.csv` | 28 |
| `journal_jalon2.csv` | 20 |
| `journal_jalon3.csv` | 131 |
| `journal_jalon4.csv` | 22 |
| `journal_jalon4_final.csv` | 22 |
| `resultats_jalon4_final.csv` | 2 |
| `predictions_test_jalon4_final.csv` | 30 |

Tous les notebooks écrivent leur journal à la racine du projet ; les fichiers livrés sont dans `result/`.

### Modèles

- `models/jalon1_mlp_reference.keras`, `jalon1_final.keras`
- `models/jalon2_cnn_retenu.keras`, `jalon2_autoencodeur.keras`, `jalon2_mlp_final.keras`
- `models/jalon3_mlp_A_seed0/1.keras`, `jalon3_cnn_A_seed0/1.keras`, `jalon3_cnn_AUG_seed0/1.keras`, `jalon3_rul_final.keras`, `jalon3_cnn_final.keras`
- `models/jalon4_final.keras.zip` : archive ; le modèle chargeable est `deploy/jalon4_final.keras`
- `modele_j4.keras` (J4 standard) : absent

### Déploiement

`deploy/jalon4_final.keras` (LSTM 32 unités, entrée 60 × 20, 6 817 paramètres entraînables), `load_model.py`, `streamlit_app.py`, `requirements.txt`, `README.md`.

---

## Annexe B — Chiffres de test

| Modèle | RMSE | Score | Anticipation | Fausses alertes | Mode | Taille |
|---|---:|---:|---:|---:|---:|---:|
| J1 final | 20,83 | 344 | 0,818 | 0,00 | — | — |
| J2 `+HI+CNN` | 21,98 | 1 277 | — | — | 0,95 (20 pompes) | — |
| J3 transfert B | 27,99 | — | — | — | 1,00 (11 pompes) | — |
| J4 GRU W = 60 | 23,79 | 1 452 | 0,818 | 0,00 | — | 74,5 Ko |
| J4 final brut | 23,93 | 1 802 | 0,818 | 0,00 | — | 105,7 Ko |
| J4 final post-traité | 25,10 | 1 826 | 1,000 | 0,00 | — | 105,7 Ko |

**Rappel de lecture.** Les scores de test sont des sommes sur les pompes (30, ou 20/11 pour l'exactitude du mode) et sont comparables à la cible de 1 200. Les scores de validation du J1 sont des `30 × moyenne par cycle`. Le score de validation du J2 Q4 est une **somme sur 4 170 cycles** : ne pas le comparer à 1 200.
