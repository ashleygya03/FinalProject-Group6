"""Unit tests for utility functions."""
import unittest
import pandas as pd
from src.utils import add_class_name_prefix

class TestUtils(unittest.TestCase):
    def test_add_class_name_prefix(self):
        """Test adding class name prefix to filename column."""
        data = {'filename': ['0_1.jpg', 'a_0.jpg', 'z_99.png']}
        df = pd.DataFrame(data)
        
        result_df = add_class_name_prefix(df, 'filename')
        
        # '0' gets '1', 'a' gets '0', 'z' gets '9' based on prefixing logic
        self.assertEqual(result_df['filename'].tolist(), ['1/0_1.jpg', '0/a_0.jpg', '9/z_99.png'])

if __name__ == '__main__':
    unittest.main()
