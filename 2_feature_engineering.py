"""
Part 2: Feature Engineering
"""

from utils import *

class CodeFeatureExtractor:
    """Extract features from code as described in the document"""
    
    def __init__(self):
        self.features = []
    
    def collect_code_structure(self, code_content):
        """Implementation of CollectCodeStructure algorithm from the document"""
        features = {
            'num_lines': 0,
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_variables': 0,
            'num_loops': 0,
            'num_conditionals': 0,
            'max_nesting_depth': 0,
            'avg_line_length': 0,
            'num_comments': 0,
            'cyclomatic_complexity': 1,
            'num_operators': 0,
            'num_operands': 0,
            'vocabulary_size': 0,
            'program_length': 0,
            'halstead_volume': 0,
            'maintainability_index': 0
        }
        
        try:
            lines = code_content.split('\n')
            features['num_lines'] = len(lines)
            features['avg_line_length'] = np.mean([len(line) for line in lines]) if lines else 0
            features['num_comments'] = sum(1 for line in lines if line.strip().startswith('#'))
            
            tree = ast.parse(code_content)
            
            class CodeVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.functions = 0
                    self.classes = 0
                    self.imports = 0
                    self.variables = 0
                    self.loops = 0
                    self.conditionals = 0
                    self.max_depth = 0
                    self.current_depth = 0
                    self.operators = 0
                    self.operands = 0
                    self.tokens = set()
                
                def visit_FunctionDef(self, node):
                    self.functions += 1
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_ClassDef(self, node):
                    self.classes += 1
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_Import(self, node):
                    self.imports += 1
                    self.generic_visit(node)
                
                def visit_ImportFrom(self, node):
                    self.imports += 1
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        self.variables += 1
                    self.tokens.add(node.id)
                    self.operands += 1
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self.loops += 1
                    self.generic_visit(node)
                
                def visit_While(self, node):
                    self.loops += 1
                    self.generic_visit(node)
                
                def visit_If(self, node):
                    self.conditionals += 1
                    self.generic_visit(node)
                
                def visit_BinOp(self, node):
                    self.operators += 1
                    self.generic_visit(node)
            
            visitor = CodeVisitor()
            visitor.visit(tree)
            
            features['num_functions'] = visitor.functions
            features['num_classes'] = visitor.classes
            features['num_imports'] = visitor.imports
            features['num_variables'] = visitor.variables
            features['num_loops'] = visitor.loops
            features['num_conditionals'] = visitor.conditionals
            features['max_nesting_depth'] = visitor.max_depth
            features['num_operators'] = visitor.operators
            features['num_operands'] = visitor.operands
            features['vocabulary_size'] = len(visitor.tokens)
            features['program_length'] = visitor.operators + visitor.operands
            
            # Halstead metrics
            if features['vocabulary_size'] > 0 and features['program_length'] > 0:
                features['halstead_volume'] = features['program_length'] * np.log2(features['vocabulary_size'] + 1)
            
            # Cyclomatic complexity (simplified)
            features['cyclomatic_complexity'] = 1 + visitor.conditionals + visitor.loops
            
            # Maintainability Index (simplified version)
            if features['halstead_volume'] > 0 and features['num_lines'] > 0:
                features['maintainability_index'] = max(0, (171 - 5.2 * np.log(features['halstead_volume']) 
                                                           - 0.23 * features['cyclomatic_complexity'] 
                                                           - 16.2 * np.log(features['num_lines'])) * 100 / 171)
            
        except Exception as e:
            print(f"Error parsing code: {e}")
        
        return features
    
    def extract_sequence_features(self, code_content):
        """Extract sequence of code (soc_f) features"""
        tokens = []
        try:
            tree = ast.parse(code_content)
            for node in ast.walk(tree):
                tokens.append(type(node).__name__)
        except:
            pass
        
        # Create n-gram features
        ngram_features = defaultdict(int)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            ngram_features[bigram] += 1
        
        return ngram_features

if __name__ == '__main__':
    # Example usage
    with open('./data/collected_code.json', 'r') as f:
        code_files = json.load(f)
    
    feature_extractor = CodeFeatureExtractor()
    
    all_features = []
    for file in code_files:
        features = feature_extractor.collect_code_structure(file['content'])
        all_features.append(features)
        
    # Save the features
    with open('./data/features.json', 'w') as f:
        json.dump(all_features, f, indent=4)
        
    print(f"Extracted features from {len(all_features)} files and saved to ./data/features.json")
