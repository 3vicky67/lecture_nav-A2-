"""
Safety Filters: Prompt Injection Detection & PII Redaction
Protects against malicious inputs and sensitive data exposure
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SafetyFilter:
    def __init__(self):
        # Prompt injection patterns
        self.injection_patterns = [
            r'ignore\s+(previous|all)\s+instructions',
            r'forget\s+(everything|all)',
            r'you\s+are\s+now\s+(a|an)',
            r'system\s*:\s*',
            r'<\|.*?\|>',
            r'\[.*?\]',
            r'\{.*?\}',
            r'role\s*:\s*(assistant|user|system)',
            r'prompt\s*:\s*',
            r'instruction\s*:\s*',
            r'override\s+',
            r'bypass\s+',
            r'jailbreak',
            r'dan\s+mode',
            r'developer\s+mode'
        ]
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'url': r'https?://[^\s<>"]+',
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Basic name pattern
        }
        
        self.redaction_placeholder = "[REDACTED]"
        
    def detect_prompt_injection(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect potential prompt injection attempts
        
        Returns:
            (is_injection, detected_patterns)
        """
        text_lower = text.lower()
        detected_patterns = []
        
        for pattern in self.injection_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detected_patterns.extend(matches)
        
        is_injection = len(detected_patterns) > 0
        
        if is_injection:
            logger.warning(f"Potential prompt injection detected: {detected_patterns}")
        
        return is_injection, detected_patterns
    
    def detect_pii(self, text: str) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Detect Personally Identifiable Information
        
        Returns:
            (has_pii, pii_by_type)
        """
        pii_found = {}
        has_pii = False
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pii_found[pii_type] = matches
                has_pii = True
        
        if has_pii:
            logger.warning(f"PII detected: {list(pii_found.keys())}")
        
        return has_pii, pii_found
    
    def redact_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redact PII from text
        
        Returns:
            (redacted_text, redaction_counts)
        """
        redacted_text = text
        redaction_counts = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, redacted_text, re.IGNORECASE)
            if matches:
                redacted_text = re.sub(pattern, self.redaction_placeholder, redacted_text, flags=re.IGNORECASE)
                redaction_counts[pii_type] = len(matches)
        
        return redacted_text, redaction_counts
    
    def sanitize_input(self, text: str) -> Dict:
        """
        Comprehensive input sanitization
        
        Returns:
            sanitization_report
        """
        report = {
            'original_text': text,
            'sanitized_text': text,
            'is_safe': True,
            'warnings': [],
            'redactions': {},
            'injection_detected': False,
            'pii_detected': False
        }
        
        # Check for prompt injection
        is_injection, injection_patterns = self.detect_prompt_injection(text)
        if is_injection:
            report['is_safe'] = False
            report['injection_detected'] = True
            report['warnings'].append(f"Prompt injection detected: {injection_patterns}")
            # Don't process potentially malicious input
            return report
        
        # Check for PII
        has_pii, pii_found = self.detect_pii(text)
        if has_pii:
            report['pii_detected'] = True
            report['warnings'].append(f"PII detected: {list(pii_found.keys())}")
            
            # Redact PII
            redacted_text, redaction_counts = self.redact_pii(text)
            report['sanitized_text'] = redacted_text
            report['redactions'] = redaction_counts
        
        return report
    
    def validate_search_query(self, query: str) -> Dict:
        """Validate search query for safety"""
        sanitization_report = self.sanitize_input(query)
        
        if not sanitization_report['is_safe']:
            return {
                'valid': False,
                'error': 'Query contains potentially malicious content',
                'details': sanitization_report['warnings']
            }
        
        return {
            'valid': True,
            'sanitized_query': sanitization_report['sanitized_text'],
            'warnings': sanitization_report['warnings']
        }
    
    def validate_ai_question(self, question: str) -> Dict:
        """Validate AI question for safety"""
        sanitization_report = self.sanitize_input(question)
        
        if not sanitization_report['is_safe']:
            return {
                'valid': False,
                'error': 'Question contains potentially malicious content',
                'details': sanitization_report['warnings']
            }
        
        return {
            'valid': True,
            'sanitized_question': sanitization_report['sanitized_text'],
            'warnings': sanitization_report['warnings']
        }

# Global safety filter instance
safety_filter = SafetyFilter()
