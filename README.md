# BeerCooler

As a data scientist, it is sometimes very hard to explain clearly to non-data-minded people what you do for living. I have friends who literally don't have a clue what I'm doing all day at the office. For those people, I'm working on a new project; a beer-cooling whisky barrel that is able to communicate with it's users about the beers that are in it. For example, when someone picks the last bottle out of the barrel, a siren should go off to trigger the host to refill the barrel with new beers. Ofcourse, a lot of other functions are possible to (an ambilight based on the beer brand, temperature checks, etc.).

<picture of barrel>

To let the barrel know what's in it, I've placed a Raspberri Pi with camera on the barrel and wrote some computer vision scripts to process the images. I wanted to make the scripts reproducable, so that you can use it to discuss the possibilities of data science/AI with your friends, too. And because I don't expect you to build a beer-booling whisky barrel yourselves, I tried to make it a little more accessible. Therefore, I created some scripts that do the magic (beer bottle detection, beer brand classification) and can easily be combined with your own phone camera.

<streamlit output>

## Structure

The repo consists of the following files:
- 

scripts uitleg
wat te doen:
- clone repo
- download IP camera
- check IP adres waarop camera gestreamd wordt
- run streamlit_camera.py script met IP adres
