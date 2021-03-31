# BLACKMED 

Black Med is a platform initiated by Invernomuto for Manifesta 12: The Planetary Garden.

Black Med is a long term research project by the artist duo Invernomuto, focusing on music culture in the Mediterranean area, which takes the form of an online platform and a series of live events.

This repository holds the code used for assigning high level music descriptors for the Black Med music database hosted on Sanity (https://www.sanity.io/) content platform.

It is practically an ML/DL pipeline using Python and Keras in order to:
 - Extract low level music signal information using the audio DSP Librosa library.
 - Train a MLP regression model with user annotated tracks on Darkness, Dynamicity and Jazzicity. The values  range in the (-1, 1) range for each high level feature.
 - Predict the above high-level semantic descriptors for new music.
 - Update the Sanity content


 
