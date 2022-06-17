"""my_dataset dataset."""

import tensorflow_datasets as tfds
from pathlib import Path 

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(names=['A', 'B']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')
    path = dl_manager.extract('dataset.zip')
    
    # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'trainA': self._generate_examples(path / 'trainA', 'A'),
        'trainB': self._generate_examples(path / 'trainB', 'B'),
        'testA' : self._generate_examples(path / 'testA', 'A'),
        'testB' : self._generate_examples(path / 'testB', 'B'),
    }

  def _generate_examples(self, path, label):
    """Yields examples."""
    # TODO(my_dataset): Yields (key, example) tuples from the dataset
    for f in path.glob('*.png'):
      key = f'{label}{Path(f).stem}'
      print(f'***{key}***')
      print(f'***{f}***')
      record = {
          'image': f,
          'label': label,
      }
      yield key, record
