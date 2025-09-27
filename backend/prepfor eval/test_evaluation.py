#!/usr/bin/env python3
"""
Test script for the evaluation framework
Demonstrates MRR@10, P95 latency, and ablation study capabilities
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def load_sample_gold_dataset():
    """Load sample gold dataset"""
    try:
        with open("sample_gold_dataset.json", "r") as f:
            dataset = json.load(f)
        
        # Add to API
        for video_id, queries in dataset.items():
            for query, timestamps in queries.items():
                response = requests.post(f"{BASE_URL}/api/evaluation/gold_dataset", 
                    json={
                        "video_id": video_id,
                        "query": query,
                        "timestamps": timestamps
                    })
                if response.status_code == 200:
                    print(f"✅ Added annotation: '{query}' → {timestamps}")
                else:
                    print(f"❌ Failed to add annotation: {response.text}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading sample dataset: {e}")
        return False

def test_mrr_evaluation(video_id):
    """Test MRR@10 evaluation"""
    print(f"\n🧪 Testing MRR@10 evaluation for video: {video_id}")
    
    response = requests.post(f"{BASE_URL}/api/evaluation/evaluate", 
        json={"video_id": video_id})
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ MRR@10 Evaluation Results:")
        print(f"   - Average MRR@10: {result['avg_mrr_at_10']:.3f}")
        print(f"   - Average Precision@K: {result['avg_precision_at_k']:.3f}")
        print(f"   - Average Latency: {result['avg_latency']:.3f}s")
        print(f"   - Meets MRR Target: {'✅' if result['meets_mrr_target'] else '❌'}")
        print(f"   - Meets Latency Target: {'✅' if result['meets_latency_target'] else '❌'}")
        print(f"   - Overall Pass: {'✅' if result['overall_pass'] else '❌'}")
        return result
    else:
        print(f"❌ MRR evaluation failed: {response.text}")
        return None

def test_latency_metrics():
    """Test P95 latency metrics"""
    print(f"\n⏱️ Testing P95 latency metrics")
    
    response = requests.get(f"{BASE_URL}/api/metrics")
    
    if response.status_code == 200:
        metrics = response.json()
        search_metrics = metrics['search_metrics']
        print(f"✅ Current Metrics:")
        print(f"   - Search Count: {search_metrics['count']}")
        print(f"   - P95 Latency: {search_metrics['p95_latency']:.3f}s")
        print(f"   - Average Latency: {search_metrics['avg_latency']:.3f}s")
        print(f"   - Meets P95 Target (≤2.0s): {'✅' if search_metrics['meets_p95_target'] else '❌'}")
        return search_metrics
    else:
        print(f"❌ Metrics retrieval failed: {response.text}")
        return None

def test_ablation_study(video_id):
    """Test ablation study"""
    print(f"\n🔬 Testing ablation study for video: {video_id}")
    
    response = requests.post(f"{BASE_URL}/api/evaluation/ablation", 
        json={
            "video_id": video_id,
            "window_sizes": [30, 45, 60],
            "overlap_ratios": [0.25, 0.33, 0.5]
        })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Ablation Study Results:")
        print(f"   - Configurations Tested: {result['configurations_tested']}")
        print(f"   - Test Queries: {len(result['test_queries'])}")
        
        print(f"\n📊 Configuration Comparison:")
        for config, data in result['results'].items():
            if 'error' in data:
                print(f"   - {config}: ❌ Error - {data['error']}")
            else:
                print(f"   - {config}: Window={data['window_size']}s, Overlap={data['overlap_seconds']}s")
                print(f"     Segments: {data['segments_count']}, Avg Latency: {data['avg_latency']:.3f}s")
        
        return result
    else:
        print(f"❌ Ablation study failed: {response.text}")
        return None

def test_full_test_suite(video_id):
    """Test complete test suite"""
    print(f"\n🎯 Running full test suite for video: {video_id}")
    
    response = requests.post(f"{BASE_URL}/api/evaluation/test_suite", 
        json={"video_id": video_id})
    
    if response.status_code == 200:
        result = response.json()
        assessment = result['overall_assessment']
        
        print(f"✅ Full Test Suite Results:")
        print(f"   - MRR Test: {'✅ PASS' if assessment['mrr_test_passed'] else '❌ FAIL'}")
        print(f"   - Latency Test: {'✅ PASS' if assessment['latency_test_passed'] else '❌ FAIL'}")
        print(f"   - All Tests: {'✅ PASS' if assessment['all_tests_passed'] else '❌ FAIL'}")
        
        return result
    else:
        print(f"❌ Full test suite failed: {response.text}")
        return None

def main():
    """Main test function"""
    print("🚀 Starting Evaluation Framework Test")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("\n❌ Cannot proceed - API is not running")
        print("Please start the server with: python server_fast.py")
        sys.exit(1)
    
    # Load sample gold dataset
    print(f"\n📊 Loading sample gold dataset...")
    if not load_sample_gold_dataset():
        print("❌ Cannot proceed - failed to load sample dataset")
        sys.exit(1)
    
    # Use sample video ID
    video_id = "sample_video_1"
    
    # Test individual components
    mrr_result = test_mrr_evaluation(video_id)
    latency_result = test_latency_metrics()
    ablation_result = test_ablation_study(video_id)
    
    # Test full suite
    suite_result = test_full_test_suite(video_id)
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 30)
    
    if mrr_result:
        print(f"MRR@10: {'✅' if mrr_result['overall_pass'] else '❌'}")
    
    if latency_result:
        print(f"P95 Latency: {'✅' if latency_result['meets_p95_target'] else '❌'}")
    
    if ablation_result:
        print(f"Ablation Study: ✅ ({ablation_result['configurations_tested']} configs tested)")
    
    if suite_result:
        assessment = suite_result['overall_assessment']
        print(f"Overall: {'✅ ALL TESTS PASS' if assessment['all_tests_passed'] else '❌ SOME TESTS FAIL'}")
    
    print(f"\n🎉 Evaluation framework test completed!")
    print(f"💡 Visit http://localhost:3000 and click 'Evaluation Dashboard' to use the UI")

if __name__ == "__main__":
    main()
