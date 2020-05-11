# Changelog for pyklopp

# Development Version

## 0.3
**Important:**
* IMPORTANT: the *config* will deprecate in subsequent versions and will be replaced by a schema-based meta-file
* introduced a first version of a meta-info-schema
* introduced a new mandatory parameter *meta* which will determine the path to the meta information file

**Misc:**
* started unifying behaviour on passing save path (we will create one config file per command call)
* we try to introduce functionality decoupled from the heavy-weight commands used to provide pyklopp as a tool

## 0.2
* Logging of evaluation metrics during training
* Sorting of configuration on standard output and when writing to file
* New keys in config such as training metrics
* Started unifying behaviour of loading modules
* automatically renaming of persistence path names to avoid overwriting issues
* changed paramter *model_path* to *model_root_path*
* gitlab CI added

## 0.1
* added commands *init* and *train*
* custom module loading for model or data set
* capturing some simple timings
* random seed initialization
* prototypical command usage in readme