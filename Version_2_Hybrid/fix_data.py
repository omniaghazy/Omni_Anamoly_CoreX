import os
import shutil
import pandas as pd
import logging

# Configure logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DataPorter:
    """
    Handles initial data ingestion, converting Excel to CSV or copying 
    existing CSV files to the designated project directory.
    """
    def __init__(self, input_path, target_folder='data/RobotArm'):
        self.input_path = input_path
        self.target_folder = target_folder
        self.target_file = os.path.join(self.target_folder, 'all_data.csv')
        self._prepare_directory()

    def _prepare_directory(self):
        """Creates the target directory if it doesn't exist."""
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
            logging.info(f"Created directory: {self.target_folder}")

    def process(self):
        """Main execution logic for file processing."""
        if not os.path.exists(self.input_path):
            logging.error(f"Input file not found: {self.input_path}")
            return False

        file_ext = os.path.splitext(self.input_path)[1].lower()

        try:
            if file_ext == '.csv':
                self._handle_csv()
            elif file_ext in ['.xlsx', '.xls']:
                self._handle_excel()
            else:
                logging.warning(f"Unsupported file extension: {file_ext}")
                return False
            
            logging.info(f"Process completed successfully. Target: {self.target_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to process data: {str(e)}")
            return False

    def _handle_csv(self):
        """Copies existing CSV to target path."""
        logging.info(f"Detected CSV. Copying {self.input_path}...")
        shutil.copy(self.input_path, self.target_file)

    def _handle_excel(self):
        """Converts Excel to CSV using Pandas for better data integrity."""
        logging.info(f"Detected Excel. Converting {self.input_path}...")
        df = pd.read_excel(self.input_path)
        df.to_csv(self.target_file, index=False, encoding='utf-8')

if __name__ == "__main__":
    # You can easily change the filename here
    RAW_DATA_SOURCE = "rtde_data.csv" 
    
    porter = DataPorter(input_path=RAW_DATA_SOURCE)
    if porter.process():
        print("\n" + "="*30)
        print("🚀 Pipeline Stage 1 Ready")
        print("Next: Run 'python data_preprocess.py'")
        print("="*30)