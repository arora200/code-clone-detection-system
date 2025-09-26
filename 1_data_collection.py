"""
Part 1: GitHub Data Collection
"""
import stat
import errno
from utils import *

def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
        func(path)
    else:
        raise

class GitHubDataCollector:
    """Collect Python CLI projects from GitHub"""
    
    def __init__(self, github_token=None):
        self.github_token = github_token
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({'Authorization': f'token {github_token}'})
    
    def search_cli_projects(self, query="python cli", max_repos=20):
        """Search for Python CLI projects on GitHub"""
        url = "https://api.github.com/search/repositories"
        params = {
            'q': f'{query} language:python stars:>50',
            'sort': 'stars',
            'per_page': max_repos
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()['items']
        else:
            print(f"Error: {response.status_code}")
            return []
    
    def clone_and_extract_code(self, repo_url, temp_dir="./data/repos"):
        """Clone repository and extract Python files"""
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(temp_dir, repo_name)
        
        try:
            if os.path.exists(repo_path):
                print(f"Repository {repo_name} already exists. Skipping clone.")
            else:
                Repo.clone_from(repo_url, repo_path)
            
            python_files = []
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            python_files.append({
                                'path': file_path,
                                'content': content,
                                'repo': repo_name
                            })
            

            return python_files
        except Exception as e:
            print(f"Error cloning {repo_url}: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    data_collector = GitHubDataCollector()
    repos = data_collector.search_cli_projects(max_repos=20)
    all_code_files = []
    for repo in repos:
        print(f"Processing repository: {repo['name']}")
        clone_url = repo['clone_url']
        code_files = data_collector.clone_and_extract_code(clone_url)
        all_code_files.extend(code_files)
    
    # Save the collected data
    with open('./data/collected_code.json', 'w') as f:
        json.dump(all_code_files, f, indent=4)
        
    print(f"Collected {len(all_code_files)} Python files and saved to ./data/collected_code.json")
