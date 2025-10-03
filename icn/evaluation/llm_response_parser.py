"""
LLM Response Parser for Malicious Package Detection
Extracts binary predictions and confidence scores from natural language LLM responses.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedPrediction:
    """Parsed prediction from LLM response."""
    is_malicious: bool
    confidence: float
    reasoning: str
    malicious_indicators: List[str]
    raw_response: str
    parse_success: bool
    parse_method: str  # json, regex, keyword, fallback


class LLMResponseParser:
    """Parser for extracting structured predictions from LLM text responses."""
    
    def __init__(self):
        # Confidence keywords for fallback parsing
        self.high_confidence_words = [
            'definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly',
            'without doubt', 'absolutely', 'very confident', 'highly likely'
        ]
        
        self.medium_confidence_words = [
            'likely', 'probably', 'appears', 'seems', 'suggests',
            'indicates', 'moderately confident', 'reasonable confidence'
        ]
        
        self.low_confidence_words = [
            'possibly', 'might', 'could be', 'uncertain', 'unclear',
            'low confidence', 'not sure', 'difficult to determine'
        ]
        
        # Malicious indicators for pattern matching
        self.malicious_keywords = [
            'malicious', 'suspicious', 'dangerous', 'harmful', 'malware',
            'trojan', 'backdoor', 'payload', 'obfuscated', 'encoded',
            'base64', 'eval', 'exec', 'subprocess', 'shell', 'download',
            'steal', 'exfiltrate', 'credential', 'password', 'token'
        ]
    
    def parse_response(self, response_text: str, model_name: str = "") -> ParsedPrediction:
        """
        Parse LLM response to extract structured prediction.
        
        Args:
            response_text: Raw text response from LLM
            model_name: Name of the model (for logging)
            
        Returns:
            ParsedPrediction with extracted information
        """
        response_text = response_text.strip()
        
        # Method 1: Try JSON parsing
        json_result = self._try_json_parsing(response_text)
        if json_result:
            return json_result
        
        # Method 2: Try regex pattern matching for JSON-like structures
        regex_result = self._try_regex_parsing(response_text)
        if regex_result:
            return regex_result
        
        # Method 3: Try keyword-based parsing
        keyword_result = self._try_keyword_parsing(response_text)
        if keyword_result:
            return keyword_result
        
        # Method 4: Fallback to heuristic parsing
        fallback_result = self._fallback_parsing(response_text)
        
        if not fallback_result.parse_success:
            logger.warning(f"Failed to parse LLM response from {model_name}: {response_text[:100]}...")
        
        return fallback_result
    
    def _try_json_parsing(self, response_text: str) -> Optional[ParsedPrediction]:
        """Try to parse response as JSON."""
        try:
            # Multiple JSON extraction strategies
            json_candidates = []

            # Strategy 1: Find complete JSON objects
            json_matches = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            for match in json_matches:
                json_candidates.append(match.group(0))

            # Strategy 2: Find simple key-value patterns that look like JSON
            simple_json_pattern = r'\{\s*"[^"]+"\s*:\s*[^}]+\}'
            simple_matches = re.finditer(simple_json_pattern, response_text, re.DOTALL)
            for match in simple_matches:
                json_candidates.append(match.group(0))

            # Strategy 3: Try to parse the entire response as JSON
            json_candidates.append(response_text.strip())

            # Try each candidate
            for json_str in json_candidates:
                try:
                    data = json.loads(json_str)

                    # Extract required fields
                    is_malicious = data.get('is_malicious', False)
                    if isinstance(is_malicious, str):
                        is_malicious = is_malicious.lower() in ['true', 'yes', '1', 'malicious']

                    confidence = float(data.get('confidence', 0.5))
                    reasoning = str(data.get('reasoning', ''))
                    indicators = data.get('malicious_indicators', [])
                    if isinstance(indicators, str):
                        indicators = [indicators]

                    return ParsedPrediction(
                        is_malicious=bool(is_malicious),
                        confidence=max(0.0, min(1.0, confidence)),  # Clamp to [0,1]
                        reasoning=reasoning,
                        malicious_indicators=indicators,
                        raw_response=response_text,
                        parse_success=True,
                        parse_method="json"
                    )
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue  # Try next candidate

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        return None
    
    def _try_regex_parsing(self, response_text: str) -> Optional[ParsedPrediction]:
        """Try regex patterns for structured data."""
        try:
            # Look for malicious/benign classification
            malicious_pattern = r'(?:is_malicious|malicious|classification)[:=\s]*(?:true|false|yes|no|malicious|benign)'
            malicious_match = re.search(malicious_pattern, response_text, re.IGNORECASE)
            
            # Look for confidence score
            confidence_pattern = r'(?:confidence|score|probability)[:=\s]*([0-9]*\.?[0-9]+)'
            confidence_match = re.search(confidence_pattern, response_text, re.IGNORECASE)
            
            if malicious_match:
                classification_text = malicious_match.group(0).lower()
                is_malicious = any(word in classification_text for word in ['true', 'yes', 'malicious'])
                
                confidence = 0.7  # Default confidence
                if confidence_match:
                    try:
                        confidence_val = float(confidence_match.group(1))
                        # Handle percentage vs probability
                        if confidence_val > 1.0:
                            confidence_val /= 100.0
                        confidence = max(0.0, min(1.0, confidence_val))
                    except ValueError:
                        pass
                
                # Extract reasoning
                reasoning = self._extract_reasoning(response_text)
                
                # Extract indicators
                indicators = self._extract_indicators(response_text)
                
                return ParsedPrediction(
                    is_malicious=is_malicious,
                    confidence=confidence,
                    reasoning=reasoning,
                    malicious_indicators=indicators,
                    raw_response=response_text,
                    parse_success=True,
                    parse_method="regex"
                )
        
        except Exception as e:
            logger.debug(f"Regex parsing failed: {e}")
        
        return None
    
    def _try_keyword_parsing(self, response_text: str) -> Optional[ParsedPrediction]:
        """Try keyword-based classification."""
        try:
            response_lower = response_text.lower()
            
            # Count malicious vs benign indicators
            malicious_score = 0
            benign_score = 0
            
            # Malicious indicators
            malicious_words = ['malicious', 'dangerous', 'suspicious', 'harmful', 'malware', 'trojan']
            for word in malicious_words:
                malicious_score += response_lower.count(word) * 2
            
            # Benign indicators  
            benign_words = ['benign', 'safe', 'legitimate', 'clean', 'normal', 'innocent']
            for word in benign_words:
                benign_score += response_lower.count(word) * 2
            
            # Strong negative indicators
            if any(phrase in response_lower for phrase in ['not malicious', 'not suspicious', 'appears safe']):
                benign_score += 3
            
            # Strong positive indicators
            if any(phrase in response_lower for phrase in ['is malicious', 'is suspicious', 'contains malware']):
                malicious_score += 3
            
            if malicious_score > 0 or benign_score > 0:
                is_malicious = malicious_score > benign_score
                
                # Calculate confidence based on score difference
                total_score = malicious_score + benign_score
                if total_score > 0:
                    confidence = max(malicious_score, benign_score) / total_score
                    confidence = max(0.5, min(0.95, confidence))  # Reasonable range
                else:
                    confidence = 0.5
                
                reasoning = self._extract_reasoning(response_text)
                indicators = self._extract_indicators(response_text)
                
                return ParsedPrediction(
                    is_malicious=is_malicious,
                    confidence=confidence,
                    reasoning=reasoning,
                    malicious_indicators=indicators,
                    raw_response=response_text,
                    parse_success=True,
                    parse_method="keyword"
                )
        
        except Exception as e:
            logger.debug(f"Keyword parsing failed: {e}")
        
        return None
    
    def _fallback_parsing(self, response_text: str) -> ParsedPrediction:
        """Fallback parsing when structured methods fail."""
        
        # Default to conservative prediction
        is_malicious = False
        confidence = 0.5
        reasoning = "Unable to parse structured response"
        indicators = []
        
        try:
            response_lower = response_text.lower()
            
            # Simple heuristic: look for key phrases
            if any(phrase in response_lower for phrase in [
                'malicious', 'dangerous', 'suspicious', 'harmful', 'malware'
            ]):
                is_malicious = True
                confidence = 0.6
                
            elif any(phrase in response_lower for phrase in [
                'benign', 'safe', 'legitimate', 'clean', 'normal'
            ]):
                is_malicious = False
                confidence = 0.6
            
            # Adjust confidence based on confidence words
            if any(word in response_lower for word in self.high_confidence_words):
                confidence = min(0.9, confidence + 0.2)
            elif any(word in response_lower for word in self.low_confidence_words):
                confidence = max(0.5, confidence - 0.2)
            
            # Try to extract some reasoning
            sentences = response_text.split('.')[:3]  # First few sentences
            reasoning = '. '.join(sentences).strip()
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            
            # Extract any technical indicators mentioned
            indicators = []
            for keyword in self.malicious_keywords:
                if keyword in response_lower and keyword not in indicators:
                    indicators.append(keyword)
        
        except Exception as e:
            logger.warning(f"Fallback parsing failed: {e}")
        
        return ParsedPrediction(
            is_malicious=is_malicious,
            confidence=confidence,
            reasoning=reasoning or "Fallback classification",
            malicious_indicators=indicators,
            raw_response=response_text,
            parse_success=False,  # Mark as unsuccessful for tracking
            parse_method="fallback"
        )
    
    def _extract_reasoning(self, response_text: str) -> str:
        """Extract reasoning/explanation from response."""
        try:
            # Look for common explanation patterns
            explanation_patterns = [
                r'(?:reasoning|explanation|analysis)[:=\s]*([^{}\n]+)',
                r'(?:because|since|due to)([^{}\n]+)',
                r'(?:the code|this package)([^{}\n]+)',
            ]
            
            for pattern in explanation_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    reasoning = match.group(1).strip()
                    if len(reasoning) > 20:  # Meaningful explanation
                        return reasoning[:300] + ("..." if len(reasoning) > 300 else "")
            
            # Fallback: use first few sentences
            sentences = response_text.split('.')[:2]
            reasoning = '. '.join(sentences).strip()
            return reasoning[:200] + ("..." if len(reasoning) > 200 else "")
        
        except Exception:
            return "Unable to extract reasoning"
    
    def _extract_indicators(self, response_text: str) -> List[str]:
        """Extract malicious indicators mentioned in response."""
        indicators = []
        response_lower = response_text.lower()
        
        for keyword in self.malicious_keywords:
            if keyword in response_lower and keyword not in indicators:
                indicators.append(keyword)
        
        # Look for specific technical terms
        technical_patterns = [
            r'base64\.decode',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.call',
            r'os\.system',
            r'__import__',
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                indicators.append(pattern.replace('\\s*\\(', '()').replace('\\', ''))
        
        return indicators[:10]  # Limit to top 10 indicators
    
    def batch_parse_responses(
        self, 
        responses: List[Tuple[str, str]]  # (response_text, model_name) pairs
    ) -> List[ParsedPrediction]:
        """Parse multiple LLM responses."""
        
        results = []
        parse_stats = {"json": 0, "regex": 0, "keyword": 0, "fallback": 0}
        
        for response_text, model_name in responses:
            parsed = self.parse_response(response_text, model_name)
            results.append(parsed)
            parse_stats[parsed.parse_method] += 1
        
        # Log parsing statistics
        total = len(responses)
        success_rate = sum(1 for r in results if r.parse_success) / total if total > 0 else 0
        
        logger.info(f"📊 LLM Response Parsing Statistics:")
        logger.info(f"   Success rate: {success_rate:.1%} ({sum(1 for r in results if r.parse_success)}/{total})")
        logger.info(f"   Parse methods: JSON={parse_stats['json']}, Regex={parse_stats['regex']}, "
                   f"Keyword={parse_stats['keyword']}, Fallback={parse_stats['fallback']}")
        
        return results


def test_parser():
    """Test the LLM response parser with various formats."""
    
    parser = LLMResponseParser()
    
    test_responses = [
        # Perfect JSON response
        '''{"is_malicious": true, "confidence": 0.95, "reasoning": "Contains base64 encoding and eval calls", "malicious_indicators": ["base64", "eval"]}''',
        
        # JSON in text
        '''Based on my analysis, this package is malicious. Here's my assessment:
        {"is_malicious": true, "confidence": 0.8, "reasoning": "Suspicious network calls", "malicious_indicators": ["curl", "subprocess"]}
        This code exhibits clear malicious behavior.''',
        
        # Structured but not JSON
        '''Classification: malicious
        Confidence: 0.85
        Reasoning: The package contains obfuscated code and makes unauthorized network requests
        Indicators: obfuscation, network requests, base64 encoding''',
        
        # Natural language response
        '''This package is definitely malicious. It contains suspicious base64 encoded payloads and makes unauthorized system calls. I'm very confident this is malware with a confidence of about 90%.''',
        
        # Benign classification
        '''This appears to be a legitimate utility package. The code is clean and performs standard mathematical operations. Classification: benign, confidence: 0.9''',
        
        # Ambiguous response
        '''The code is somewhat suspicious but might be legitimate. Hard to say for certain. Could be malicious but not sure.'''
    ]
    
    print("🧪 Testing LLM Response Parser")
    print("=" * 50)
    
    for i, response in enumerate(test_responses, 1):
        print(f"\nTest {i}:")
        print(f"Response: {response[:100]}...")
        
        parsed = parser.parse_response(response, f"test_model_{i}")
        
        print(f"✅ Parsed Result:")
        print(f"   Malicious: {parsed.is_malicious}")
        print(f"   Confidence: {parsed.confidence:.3f}")
        print(f"   Method: {parsed.parse_method}")
        print(f"   Success: {parsed.parse_success}")
        print(f"   Reasoning: {parsed.reasoning[:80]}...")
        print(f"   Indicators: {parsed.malicious_indicators}")


if __name__ == "__main__":
    test_parser()