"""
Impossible Jobs Detection: Identify job postings with unrealistic experience requirements.
Example: Requiring 10 years of experience in LLM when transformers came out 8 years ago.
"""

import re
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Technology release dates (years)
TECHNOLOGY_DATES = {
    # LLM and Transformer-related
    'transformer': 2017,
    'bert': 2018,
    'gpt': 2018,
    'gpt-2': 2019,
    'gpt-3': 2020,
    'gpt-4': 2023,
    'llm': 2018,  # Large Language Models
    'large language model': 2018,
    'chatgpt': 2022,
    'claude': 2023,
    'llama': 2023,
    
    # Deep Learning frameworks
    'tensorflow': 2015,
    'pytorch': 2016,
    'keras': 2015,
    
    # Cloud platforms
    'aws lambda': 2014,
    'azure functions': 2016,
    'google cloud functions': 2016,
    'kubernetes': 2014,
    'docker': 2013,
    
    # Modern web frameworks
    'react': 2013,
    'vue.js': 2014,
    'angular': 2010,  # AngularJS, but modern Angular is 2016
    'next.js': 2016,
    'nuxt.js': 2016,
    
    # Mobile
    'swift': 2014,
    'kotlin': 2011,  # But Android Kotlin support is 2017
    'flutter': 2017,
    'react native': 2015,
    
    # Data Science
    'scikit-learn': 2010,
    'pandas': 2008,
    'numpy': 2006,
    'pytorch': 2016,
    'tensorflow': 2015,
    
    # Add more technologies as needed
}


class ImpossibleJobsDetector:
    """Detect impossible job requirements based on technology release dates."""
    
    def __init__(self, technology_dates: Dict[str, int] = None):
        """
        Initialize detector.
        
        Args:
            technology_dates: Dictionary mapping technology names to release years
        """
        self.technology_dates = technology_dates or TECHNOLOGY_DATES
        # Automatically use current year instead of hardcoding
        from datetime import datetime
        self.current_year = datetime.now().year
        
    def extract_experience_requirements(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract experience requirements from text.
        
        Args:
            text: Job description text
            
        Returns:
            List of tuples (technology, years_required)
        """
        if pd.isna(text):
            return []
        
        text_lower = text.lower()
        requirements = []
        
        # Pattern to match "X years of experience in/with Y" (handles 12+, 12-15, 12, etc.)
        # Patterns ordered by priority - more specific patterns first
        patterns = [
            # "X years with/using Y" - most direct pattern
            r'(\d+)\+?\s*years?\s*(?:working with|with|using|of)\s+([a-z][a-z\s\-\.]+?)(?:\s+and|\s+or|\.|,|;|$|\n)',
            # "X years Y experience/development"
            r'(\d+)\+?\s*years?\s+([a-z][a-z\s\-\.]+?)\s+(?:experience|development)',
            # "X years of experience with Y"
            r'(\d+)\+?\s*years?\s*of\s+experience\s+(?:with|in|using)\s+([a-z][a-z\s\-\.]+?)(?:\s+and|\s+or|\.|,|;|$|\n)',
            # General fallback
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience\s+)?(?:in|with)?\s*([a-z][a-z\s\-\.]+?)(?:\s+and|\.|,|;|$|\n)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                years = int(match.group(1))
                tech = match.group(2).strip()
                
                # Clean up technology name: remove trailing punctuation and extra spaces
                tech = re.sub(r'[^\w\s\-\.]', '', tech).strip()
                
                # Skip empty or very short tech names
                if len(tech) >= 2:
                    requirements.append((tech, years))
        
        return requirements
    
    def check_technology_age(self, technology: str, years_required: int) -> Tuple[bool, int]:
        """
        Check if required years of experience exceed technology age.
        
        Args:
            technology: Technology name
            years_required: Years of experience required
            
        Returns:
            Tuple of (is_impossible, technology_age)
        """
        # Normalize technology name for better matching
        tech_normalized = technology.lower().strip()
        
        # Check for exact matches first
        if tech_normalized in self.technology_dates:
            tech_age = self.current_year - self.technology_dates[tech_normalized]
            return years_required > tech_age, tech_age
        
        # Check for partial matches (e.g., "transformer" in "transformer models")
        # Also check with spaces replaced by hyphens and vice versa
        tech_variants = [
            tech_normalized,
            tech_normalized.replace(' ', '-'),
            tech_normalized.replace('-', ' '),
            tech_normalized.replace('.', '')
        ]
        
        for tech_name, release_year in self.technology_dates.items():
            for variant in tech_variants:
                if tech_name in variant or variant in tech_name:
                    tech_age = self.current_year - release_year
                    return years_required > tech_age, tech_age
        
        # If technology not found, return None (unknown)
        return None, None
    
    def detect_impossible_requirements(self, text: str) -> Dict:
        """
        Detect impossible requirements in job posting.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with detection results
        """
        requirements = self.extract_experience_requirements(text)
        impossible_requirements = []
        possible_requirements = []
        unknown_requirements = []
        
        for tech, years in requirements:
            # Skip if technology name is too short or contains only special characters
            if len(tech.strip()) < 3 or not any(c.isalpha() for c in tech):
                continue
                
            is_impossible, tech_age = self.check_technology_age(tech, years)
            
            # Only add if we found a matching technology
            if is_impossible is None:
                unknown_requirements.append({
                    'technology': tech,
                    'years_required': years,
                    'status': 'unknown'
                })
            elif is_impossible:
                impossible_requirements.append({
                    'technology': tech,
                    'years_required': years,
                    'technology_age': tech_age,
                    'status': 'impossible'
                })
            else:
                possible_requirements.append({
                    'technology': tech,
                    'years_required': years,
                    'technology_age': tech_age,
                    'status': 'possible'
                })
        
        return {
            'has_impossible_requirements': len(impossible_requirements) > 0,
            'impossible_count': len(impossible_requirements),
            'impossible_requirements': impossible_requirements,
            'possible_requirements': possible_requirements,
            'unknown_requirements': unknown_requirements,
            'total_requirements_found': len(requirements)
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'description') -> pd.DataFrame:
        """
        Analyze entire dataframe for impossible requirements.
        
        Args:
            df: DataFrame with job postings
            text_column: Name of column containing job descriptions
            
        Returns:
            DataFrame with added columns for impossible job detection
        """
        logger.info("Analyzing job postings for impossible requirements...")
        
        results = df[text_column].apply(self.detect_impossible_requirements)
        
        df['has_impossible_requirements'] = [r['has_impossible_requirements'] for r in results]
        df['impossible_requirements_count'] = [r['impossible_count'] for r in results]
        df['total_requirements_found'] = [r['total_requirements_found'] for r in results]
        df['impossible_requirements_details'] = [r['impossible_requirements'] for r in results]
        
        logger.info(f"Found {df['has_impossible_requirements'].sum()} job postings with impossible requirements")
        
        return df


def main():
    """Example usage."""
    detector = ImpossibleJobsDetector()
    
    # Example job posting
    example_text = """
    We are looking for a Senior ML Engineer with:
    - 10 years of experience in Large Language Models (LLM)
    - 8 years of experience with GPT-4
    - 5 years of experience with TensorFlow
    - 3 years of experience with React
    """
    
    result = detector.detect_impossible_requirements(example_text)
    print("Detection Results:")
    print(f"Has impossible requirements: {result['has_impossible_requirements']}")
    print(f"Impossible requirements: {result['impossible_requirements']}")
    print(f"Possible requirements: {result['possible_requirements']}")


if __name__ == "__main__":
    main()

