The code is not well documented as I currently don't have the time to add proper documentation. I wanted to upload this project anyways now as I don't know when I ever find the time to polish it up.
Everything should roughly work.

The project features:

- A GUI to view adjust and render JunoCam images to files

- Shader programs for all the rendering from the raw data

- A downloader which automatically downloads all the images with metadata, as well as the spacecraft data for a specific Perijove (It just grabs everything from the NASA websites)

- Multiple images from one Perijove can be selected at once to be blended together to cover a bigger surface of jupiter

Add a folder inside the repository named `/Data`, the downloader will put all needed files in there. As the images are downloaded in `.zip` archives, the downloader will create a temporary repository `tmp` inside `/Data`, from which the archives are extracted.

A small how-to-use is given in the drop-down menu `Extras`


Have fun! :)
