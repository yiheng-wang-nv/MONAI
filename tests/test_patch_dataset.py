# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from torch.utils.data.dataloader import DataLoader

from monai.data.patch_dataset import PatchDataset


class TestPatchDataset(unittest.TestCase):

    def test_shape(self):
        test_dataset = ['vwxyz', 'hello', 'world']
        n_per_image = len(test_dataset[0])
        result = PatchDataset(dataset=test_dataset, patch_func=lambda x: x, samples_per_image=n_per_image)

        output = []
        for item in DataLoader(result, batch_size=3, num_workers=5):
            output.append(''.join(item))
        expected = ['vwx', 'yzh', 'ell', 'owo', 'rld']
        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main()
