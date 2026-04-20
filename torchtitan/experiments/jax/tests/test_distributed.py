"""Tests for the shared sharding-map matcher in jax/distributed.py.

The matcher is used by both jax/ and torchax/ experiments. These tests
exercise it against both style conventions:
  - jax: '/' separator, `*` in map keys (post-integer-replacement form)
  - torchax: '.' separator, regex patterns (e.g. ``r'.*foo\\.weight'``)
"""

from absl.testing import absltest
from absl.testing import parameterized

from torchtitan.experiments.jax.distributed import _match_path
from torchtitan.experiments.jax.distributed import _process_sharding_name
from torchtitan.experiments.jax.llama3.sharding import sharding_map_original as jax_llama3_original
from torchtitan.experiments.jax.llama3.sharding import sharding_map_scan as jax_llama3_scan
from torchtitan.experiments.torchax.redacted.sharding import sharding_map_original as torchax_redacted_original


class ProcessShardingNameTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('dot_separator', 'layers.0.attention.wq', 'layers.*.attention.wq'),
      ('slash_separator', 'layers/0/attention/wq', 'layers/*/attention/wq'),
      ('triple_underscore', 'params___layers___0___attention', 'params___layers___*___attention'),
      ('multiple_ints', 'a/0/b/12/c', 'a/*/b/*/c'),
      ('no_ints', 'tok_embeddings/embedding', 'tok_embeddings/embedding'),
      ('leading_digit_word_not_replaced', 'norm2/weight', 'norm2/weight'),
      ('empty', '', ''),
  ])
  def test_process(self, input_name, expected):
    self.assertEqual(_process_sharding_name(input_name), expected)


class MatchPathExactTest(absltest.TestCase):
  """Exact-match path, no preprocessing needed."""

  def test_hit(self):
    m = {'norm/weight': ('fsdp',), 'freqs_cis': ()}
    self.assertEqual(_match_path('norm/weight', m), ('fsdp',))
    self.assertEqual(_match_path('freqs_cis', m), ())

  def test_miss(self):
    m = {'norm/weight': ('fsdp',)}
    self.assertIsNone(_match_path('nonexistent', m))


class MatchPathIntPreprocessingTest(absltest.TestCase):
  """jax-style paths: integer index becomes '*' for exact match."""

  def test_slash_path(self):
    m = {'layers/*/attention/wq/kernel': ('fsdp', None)}
    self.assertEqual(
        _match_path('layers/0/attention/wq/kernel', m), ('fsdp', None)
    )
    self.assertEqual(
        _match_path('layers/42/attention/wq/kernel', m), ('fsdp', None)
    )

  def test_dot_path(self):
    m = {'model.layers.*.attention.weight': ('tp', 'fsdp')}
    self.assertEqual(
        _match_path('model.layers.7.attention.weight', m), ('tp', 'fsdp')
    )

  def test_multiple_indices(self):
    m = {'a/*/b/*/c': ('fsdp',)}
    self.assertEqual(_match_path('a/3/b/11/c', m), ('fsdp',))


class MatchPathRegexTest(absltest.TestCase):
  """torchax-style paths: regex patterns in map keys."""

  def test_dot_star_prefix(self):
    m = {r'.*embedding\.weight': ('fsdp', 'tp')}
    self.assertEqual(
        _match_path('model.embedding.weight', m), ('fsdp', 'tp')
    )
    self.assertEqual(_match_path('embedding.weight', m), ('fsdp', 'tp'))

  def test_anchored_pattern(self):
    m = {r'^model\.output_transform\.weight$': ('tp', 'fsdp')}
    self.assertEqual(
        _match_path('model.output_transform.weight', m), ('tp', 'fsdp')
    )
    # Anchored: extra prefix should not match.
    self.assertIsNone(
        _match_path('something.model.output_transform.weight', m)
    )

  def test_alternation(self):
    m = {r'.*(wq|wk|wv)\.weight': ('fsdp',)}
    self.assertEqual(_match_path('layer.wq.weight', m), ('fsdp',))
    self.assertEqual(_match_path('layer.wk.weight', m), ('fsdp',))
    self.assertIsNone(_match_path('layer.wo.weight', m))


class MatchPathPrecedenceTest(absltest.TestCase):
  """Exact match wins over preprocessed match wins over regex."""

  def test_exact_wins_over_preprocessed(self):
    m = {
        'layers/0/attention': ('A',),
        'layers/*/attention': ('B',),
    }
    self.assertEqual(_match_path('layers/0/attention', m), ('A',))

  def test_preprocessed_wins_over_regex(self):
    m = {
        'layers/*/attention': ('B',),
        r'.*attention': ('C',),
    }
    self.assertEqual(_match_path('layers/0/attention', m), ('B',))


class MatchPathInvalidRegexTest(absltest.TestCase):
  """Map keys that are not valid regex should be skipped, not raise."""

  def test_invalid_regex_ignored(self):
    m = {'[unclosed': ('bad',), 'fallback': ('good',)}
    # Non-matching path shouldn't blow up on the invalid pattern.
    self.assertIsNone(_match_path('something', m))
    # Exact match still works.
    self.assertEqual(_match_path('fallback', m), ('good',))


class RealShardingMapTest(absltest.TestCase):
  """Regression guard: real sharding maps from jax/ and torchax/ resolve."""

  def test_jax_llama3_non_scan(self):
    self.assertEqual(
        _match_path('tok_embeddings/embedding', jax_llama3_original), ()
    )
    self.assertEqual(
        _match_path('layers/0/attention/wq/kernel', jax_llama3_original),
        ('fsdp', None),
    )
    self.assertEqual(
        _match_path('layers/3/feed_forward/w2/kernel', jax_llama3_original),
        (None, 'fsdp'),
    )
    self.assertEqual(
        _match_path('norm/weight', jax_llama3_original), ('fsdp',)
    )

  def test_jax_llama3_scan(self):
    # Scan stacks params, so paths have no integer index.
    self.assertEqual(
        _match_path('layers/attention_wq_kernel', jax_llama3_scan),
        (None, 'fsdp', None),
    )
    self.assertEqual(_match_path('freqs_cis', jax_llama3_scan), ())

  def test_torchax_redacted(self):
    self.assertEqual(
        _match_path(
            'model.layers.0.attention.qkv_transform.wrapped.fused_linear.weight',
            torchax_redacted_original,
        ),
        ('tp', 'fsdp'),
    )
    self.assertEqual(
        _match_path('model.embedding.weight', torchax_redacted_original),
        ('fsdp', 'tp'),
    )
    self.assertEqual(
        _match_path('model.output_transform.weight', torchax_redacted_original),
        ('tp', 'fsdp'),
    )
    self.assertEqual(
        _match_path('layer.norm.weight', torchax_redacted_original), ('fsdp',)
    )


if __name__ == '__main__':
  absltest.main()
