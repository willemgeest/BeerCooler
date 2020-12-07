# BeerCooler

As a data scientist, it is sometimes very hard to explain clearly to non-data-minded people what you do for living. I have friends who literally don't have a clue what I'm doing all day at the office. For those people, I'm working on a new project; a beer-cooling barrel that is able to communicate with it's users about the beers that are in it. For example, when someone picks the last bottle out of the barrel, a siren should go off to trigger the host to refill the barrel with new beers. Ofcourse, a lot of other functions are possible to (an ambilight based on the beer brand, temperature checks, etc.).

To let the barrel know what's in it, I've placed a Raspberri Pi with camera on the barrel and wrote some computer vision scripts to process the images. These scripts can be found in this repo. 
