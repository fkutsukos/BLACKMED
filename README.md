# BLACKMED 

Black Med is a platform initiated by Invernomuto for Manifesta 12: The Planetary Garden https://blackmed.invernomuto.info/.

Black Med is a long term research project by the artist duo Invernomuto, focusing on music culture in the Mediterranean area, which takes the form of an online platform and a series of live events.

This repository is practically a deep learning pipeline using Python and Keras in order train a model for predicting high level music descriptors for the Black Med music database hosted on Sanity (https://www.sanity.io/) content platform.

## Features:
 - Extract low level music signal information using the audio DSP Librosa (https://librosa.org/doc/latest/index.html) and Scaper (https://github.com/justinsalamon/scaper) libraries.
 - Train a MLP regression model with user annotated tracks on Darkness, Dynamicity and Jazzicity. The values range in the (-1, 1) range for each high level feature.
 - The MLP model is used to predict the above high-level semantic descriptors for new music content added to Black Med.
 - Update the Sanity content database with the high level descriptors.

## Help / Reporting Bugs
Email fotis.kutsukos@gmail.com
