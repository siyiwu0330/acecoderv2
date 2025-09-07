# AceCoderV2 Changelog

## Version 2.1.0 (2024-09-05)

### 🚀 New Features

#### 1. Skip Step4 Functionality
- **Added `--skip_step4` parameter** to `main.py` for controlling cross-round evaluation
- **Gradio interface enhancement** with skip step4 checkbox option
- **Performance optimization** for large datasets to prevent hanging issues
- **Backward compatibility** maintained - default behavior unchanged

#### 2. Parquet Conversion Tool
- **New `convert_to_parquet.py`** script for efficient data format conversion
- **Based on `hf_dataset_converter.py`** logic for data extraction and processing
- **Complete merge functionality** for multi-round data consolidation
- **Enhanced data validation** and error handling
- **Comprehensive statistics** and metadata preservation

### 🔧 Technical Improvements

#### Pipeline Enhancements
- **Smart data extraction** from `synthesis_result` for generated problems and test cases
- **Intelligent deduplication** with cross-round merging capabilities
- **Flexible filtering** by round, minimum test cases, and other criteria
- **Robust error handling** and validation throughout the pipeline

#### Docker Integration
- **Updated Dockerfile** with version 2.1.0 information
- **Enhanced build process** with all new dependencies included
- **Improved documentation** and usage examples
- **Docker Hub deployment** with both `latest` and `2.1.0` tags

### 📊 Data Processing Features

#### Convert to Parquet Tool
```bash
# Basic conversion
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output

# Advanced usage with filtering
python convert_to_parquet.py \
    --jsonl_path data.jsonl \
    --local_dir output \
    --target_round 3 \
    --min_tests 5 \
    --test_size 1000 \
    --stats
```

#### Skip Step4 Usage
```bash
# Skip step4 for large datasets
python main.py --output_dir outputs/test --rounds 2 --max_samples 100 --skip_step4 True

# Use Gradio interface with skip step4 option
python integrated_gradio_app.py
```

### 🎯 Performance Benefits

#### Skip Step4 Impact
- **Solves hanging issues** with large problem sets (>50 problems)
- **Faster completion** by skipping computationally expensive cross-round evaluation
- **Memory efficient** with reduced resource usage
- **Stable execution** without timeouts or process blocking

#### Parquet Format Advantages
- **Storage efficiency** - more compact than JSON format
- **Faster loading** - columnar storage for better performance
- **Memory friendly** - supports chunked reading for large datasets
- **Tool compatibility** - works seamlessly with pandas, Dask, etc.

### 📁 New Files Added

1. **`convert_to_parquet.py`** - Main parquet conversion tool
2. **`PARQUET_CONVERSION_USAGE.md`** - Detailed usage documentation
3. **`SKIP_STEP4_USAGE.md`** - Skip step4 functionality guide
4. **`CHANGELOG.md`** - This changelog file

### 🔄 Backward Compatibility

- **All existing functionality** remains unchanged
- **Default parameters** maintain previous behavior
- **API compatibility** preserved across all interfaces
- **Data format** remains consistent with previous versions

### 🐳 Docker Hub Availability

The new version is now available on Docker Hub:

```bash
# Pull the latest version
docker pull siyiwu0330/acecoderv2:latest

# Or pull specific version
docker pull siyiwu0330/acecoderv2:2.1.0

# Run the container
docker run -d -p 7860:7860 --name acecoderv2 siyiwu0330/acecoderv2:latest
```

### 📈 Usage Statistics

- **Docker image size**: 8.94GB
- **Python version**: 3.11
- **Dependencies**: All AI/ML packages pre-installed
- **Features**: Complete adversarial generation pipeline + new tools

### 🎉 Summary

Version 2.1.0 introduces significant improvements for handling large datasets and data processing workflows. The skip step4 functionality solves critical performance issues, while the parquet conversion tool provides efficient data management capabilities. All changes maintain full backward compatibility while adding powerful new features for advanced users.

---

**Docker Hub**: https://hub.docker.com/r/siyiwu0330/acecoderv2
**GitHub**: [Project Repository]
**Documentation**: See individual usage guides for detailed instructions

