import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

class RunoffDiagnostics:
    """Class for diagnosing runoff calculations with logging support"""
    
    def __init__(self, logger):
        """
        Initialize diagnostics with logger
        
        Args:
            logger: ParameterLogger instance
        """
        self.logger = logger

    def collect_diagnostics(self, domain, date: datetime) -> Dict[str, Any]:
        """
        Collect diagnostic information about runoff calculations
        
        Args:
            domain: ModelDomain instance
            date: Current simulation date
            
        Returns:
            dict: Diagnostic information
        """
        diagnostics = {
            'timestamp': date.strftime('%Y-%m-%d'),
            'precipitation': self._collect_precip_diagnostics(domain),
            'runoff_module': self._collect_module_diagnostics(domain),
            'curve_numbers': self._collect_cn_diagnostics(domain),
            'antecedent_conditions': self._collect_antecedent_diagnostics(domain),
            'landuse_stats': self._collect_landuse_diagnostics(domain)
        }
        
        # Add parameter details for each landuse type
        if domain.runoff_module is not None and domain.runoff_module.params:
            diagnostics['landuse_parameters'] = self._collect_landuse_params(domain)
        
        # Add value counts
        if domain.runoff_module is not None:
            diagnostics['value_counts'] = self._collect_value_counts(domain)
            
        return diagnostics

    def _collect_precip_diagnostics(self, domain) -> Dict[str, Any]:
        """Collect precipitation-related diagnostics"""
        return {
            'gross_precip_stats': self._get_array_stats(domain.gross_precip),
            'rainfall_stats': self._get_array_stats(domain.rainfall),
            'net_rainfall_stats': self._get_array_stats(domain.net_rainfall)
        }

    def _collect_module_diagnostics(self, domain) -> Dict[str, Any]:
        """Collect runoff module configuration diagnostics"""
        if domain.runoff_module is not None:
            return {
                'initialized': True,
                'method': domain.runoff_module.method,
                'landuse_indices_set': domain.runoff_module.landuse_indices is not None,
                'params_count': len(domain.runoff_module.params)
            }
        return {'initialized': False}

    def _collect_cn_diagnostics(self, domain) -> Dict[str, Any]:
        """Collect curve number diagnostics"""
        if domain.runoff_module is not None:
            return {
                'cn_current_stats': self._get_array_stats(domain.runoff_module.cn_current),
                'cn_normal_stats': self._get_array_stats(domain.runoff_module.cn_normal)
            }
        return {}

    def _collect_antecedent_diagnostics(self, domain) -> Dict[str, Any]:
        """Collect antecedent condition diagnostics"""
        if domain.runoff_module is not None:
            return {
                'prev_5day_precip_stats': self._get_array_stats(domain.runoff_module.prev_5day_precip)
            }
        return {}

    def _collect_landuse_diagnostics(self, domain) -> Dict[str, Any]:
        """Collect landuse-related diagnostics"""
        if domain.landuse_indices is not None:
            return {
                'unique_values': np.unique(domain.landuse_indices).tolist()
            }
        return {}

    def _collect_landuse_params(self, domain) -> Dict[int, Dict[str, float]]:
        """Collect landuse parameter diagnostics"""
        return {
            landuse_id: {
                'curve_number': params.curve_number,
                'initial_abstraction_ratio': params.initial_abstraction_ratio,
                'depression_storage': params.depression_storage,
                'impervious_fraction': params.impervious_fraction
            }
            for landuse_id, params in domain.runoff_module.params.items()
        }

    def _collect_value_counts(self, domain) -> Dict[str, int]:
        """Collect value count diagnostics"""
        return {
            'runoff_gt_zero': int(np.sum(domain.runoff_module.runoff > 0)),
            'precip_gt_zero': int(np.sum(domain.gross_precip > 0)) if domain.gross_precip is not None else 0,
            'cn_gt_zero': int(np.sum(domain.runoff_module.cn_current > 0))
        }

    @staticmethod
    def _get_array_stats(arr: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate statistics for an array"""
        if arr is None:
            return {'min': None, 'max': None, 'mean': None, 'non_zero': None}
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'non_zero': int(np.sum(arr > 0))
        }

    def log_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """
        Log diagnostic information using the parameter logger
        
        Args:
            diagnostics: Dictionary of diagnostic information
        """
        self.logger.info(f"\n=== Runoff Diagnostics for {diagnostics['timestamp']} ===")
        
        # Log module configuration
        module_info = diagnostics['runoff_module']
        self.logger.info("\nModule Configuration:")
        self.logger.info(f"Initialized: {module_info['initialized']}")
        if module_info['initialized']:
            self.logger.info(f"Method: {module_info['method']}")
            self.logger.info(f"Landuse indices set: {module_info['landuse_indices_set']}")
            self.logger.info(f"Number of parameter sets: {module_info['params_count']}")

        # Log precipitation statistics
        precip_stats = diagnostics['precipitation']
        self.logger.info("\nPrecipitation Statistics:")
        for key, stats in precip_stats.items():
            if stats['mean'] is not None:
                self.logger.info(f"{key}: Min={stats['min']:.4f}, Max={stats['max']:.4f}, "
                               f"Mean={stats['mean']:.4f}, Non-zero cells={stats['non_zero']}")

        # Log curve number statistics
        if 'curve_numbers' in diagnostics:
            cn_stats = diagnostics['curve_numbers']
            self.logger.info("\nCurve Number Statistics:")
            for key, stats in cn_stats.items():
                if stats['mean'] is not None:
                    self.logger.info(f"{key}: Min={stats['min']:.1f}, Max={stats['max']:.1f}, "
                                   f"Mean={stats['mean']:.1f}, Non-zero cells={stats['non_zero']}")

        # Log landuse information
        if 'landuse_stats' in diagnostics:
            self.logger.info("\nLanduse Information:")
            self.logger.info(f"Unique landuse values: {diagnostics['landuse_stats']['unique_values']}")

        # Log landuse parameters
        if 'landuse_parameters' in diagnostics:
            self.logger.info("\nLanduse Parameters:")
            for landuse_id, params in diagnostics['landuse_parameters'].items():
                self.logger.info(f"\nLanduse ID {landuse_id}:")
                for param_name, value in params.items():
                    self.logger.info(f"  {param_name}: {value}")

        # Log value counts
        if 'value_counts' in diagnostics:
            counts = diagnostics['value_counts']
            self.logger.info("\nValue Counts:")
            self.logger.info(f"Cells with runoff > 0: {counts['runoff_gt_zero']}")
            self.logger.info(f"Cells with precipitation > 0: {counts['precip_gt_zero']}")
            self.logger.info(f"Cells with curve number > 0: {counts['cn_gt_zero']}")
