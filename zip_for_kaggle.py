import os
import zipfile

def zip_project(source_dir, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                # Exclude .git, __pycache__, and .vscode if they exist
                if any(x in root for x in ['.git', '__pycache__', '.vscode']):
                    continue
                
                file_path = os.path.join(root, file)
                # Create the internal path using forward slashes
                arcname = os.path.relpath(file_path, source_dir).replace('\\', '/')
                zipf.write(file_path, arcname)
                print(f"Added: {arcname}")

if __name__ == "__main__":
    source = r"X:\Omnia_CoreX\Omnia_Anomaly_Detection_coreX-main"
    output = r"X:\Omnia_CoreX\coreX_project_kaggle.zip"
    zip_project(source, output)
    print(f"\n✅ Created: {output}")
