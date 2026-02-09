# Deterministic Audit Script

import os
import hashlib
import json

class QARepoAudit:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.audit_results = {}

    def calculate_file_hash(self, file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def audit_files(self):
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_hash = self.calculate_file_hash(file_path)
                self.audit_results[file_path] = file_hash

    def save_audit_results(self, output_file):
        with open(output_file, 'w') as f:
            json.dump(self.audit_results, f, indent=4)

if __name__ == '__main__':
    audit = QARepoAudit(repo_path='path/to/your/repo')  # Change this to your repo path
    audit.audit_files()
    audit.save_audit_results(output_file='audit_results.json')  # Change this to your desired output file path
