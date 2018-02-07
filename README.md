

# Camera model identification

Finest camera identification in town!

# Schedule

- Feb1:
  - Fetch files & create github repo
  - Research domain knowledge about cameras
  - Research discussions / Papers on arxiv / Blogs / Other datasets


## RAW Dataset download

Download to your project directory, add it, and commit.

```
# Kaggle dataset download
sudo pip install kaggle-cli
cd camera-id
kg download -u <username> -p <password> -c sp-society-camera-model-identification
mkdir files data
mv *.zip files/ && cd files
unzip \*.zip 
mv test ../data/
mv train ../data/
```

## Derivative datasets generation
```
# Install requirements
sudo apt-get install libturbojpeg
sudo pip3 install cffi tqdm jpeg4py
```
## Usage


# Credits

Released under the [MIT License].<br>
Authored and maintained by Dušan Josipović.

> Blog [dulex123.github.io](http://dulex123.github.io) &nbsp;&middot;&nbsp;
> GitHub [@dulex123](https://github.com/dulex123) &nbsp;&middot;&nbsp;
> Twitter [@josipovicd](https://twitter.com/josipovicd)

[MIT License]: http://mit-license.org/
