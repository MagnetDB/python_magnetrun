# Documentation des messages du fichier LogAcqNet

## 1. Messages opérationnels : démarrage / arrêt

**Acquisition_démarrée_sur_Groupe X_aimant_Y**\
Indique le début d'un cycle d'acquisition pour un groupe de capteurs.

**Acquisition_arrêtée_sur_Groupe X_aimant_Y**\
Signale l'arrêt du cycle d'acquisition, volontaire ou forcé (souvent
suite à une erreur DAQ).

------------------------------------------------------------------------

## 2. Messages de défauts détectés

### SpikeAimant

Détection d'un spike de courant anormal dans un ou plusieurs capteurs
(internes ou externes).

### DefautNums_Ax_I\_MAX

Défauts générés suite à un trigger matériel de l'installation (relais)

### Courants50Hz

Détection d'un courant 50 Hz anormalement élevé (perturbation, bruit
réseau, surtension).

------------------------------------------------------------------------

## 3. Création de fichiers TDMS

-   **Fichiers_Spike** -- Enregistrements d'événements de type spike sur tensions aimants 1s
-   **Fichiers_Default** -- Sauvegarde des défauts numériques ou 50 Hz, 60s
-   **Fichiers_Archive** -- Archive 120 Hz des données
-   **Fichiers_stats** -- Statistiques sur les données 4800 Hz, réduite à 1 Hz
-   **Overview** -- Archives aggrégées et réduites par moyennage sur le run complet, 1Hz

------------------------------------------------------------------------

## 4. Erreurs DAQ (LabVIEW / NI-DAQmx)
Souvent générée à l'arrêt, non significatif

### Error -200279

L'application ne lit pas les données assez vite → débordement du
buffer.\
**Solutions :** augmenter la taille du buffer, lire plus souvent, lire
un nombre fixe d'échantillons.

### Error -200019

Conversion ADC commencée avant la fin de la précédente.\
**Solutions :** augmenter la période entre conversions, vérifier
l'horloge externe, vérifier les boîtiers ENET.

------------------------------------------------------------------------

## 5. Tests matériel ENET / alimentations

Le système vérifie la présence et la réponse des boîtiers ENET (A1, A2,
A3, HT1, etc.).\
Message : **« Vérification boitiers PS ok »** → alimentations
opérationnelles.

------------------------------------------------------------------------

## 6. Messages internes au système

### Enregistrement du défaut...

Création d'un fichier TDMS du défaut détecté.

### Archivage périodique

Création automatique des fichiers *Archive*, *Stats* et *Overview*.

------------------------------------------------------------------------

## Résumé simplifié

-   Démarrage / Arrêt : gestion des cycles d'acquisition\
-   Default detected : détection d'anomalies (spike, I_MAX, 50 Hz)\
-   Fichiers créés : sauvegarde automatique\
-   Erreurs DAQ : problèmes de performance ou d'ADC\
-   Tests matériel : vérification ENET / alimentations\
-   Archivage interne : maintenance automatisée
