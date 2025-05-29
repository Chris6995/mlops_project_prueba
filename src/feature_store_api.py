from typing import Optional, List
from dataclasses import dataclass

import hsfs
import hopsworks

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class FeatureGroupConfig:
    name: str
    version: int
    description: str
    primary_key: List[str]
    event_time: str
    online_enabled: Optional[bool] = False

@dataclass
class FeatureViewConfig:
    name: str
    version: int
    feature_group: FeatureGroupConfig

def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    return project.get_feature_store()

# TODO: remove this function, and use get_or_create_feature_group instead
def get_feature_group(
    name: str,
    version: Optional[int] = 1
    ) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given
    feature group `name`

    Args:
        name (str): name of the feature group
        version (Optional[int], optional): _description_. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    return get_feature_store().get_feature_group(
        name=name,
        version=version,
    )

def get_or_create_feature_group(
    feature_group_metadata: FeatureGroupConfig
) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given
    feature group `name`

    Args:
        name (str): name of the feature group
        version (Optional[int], optional): _description_. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    return get_feature_store().get_or_create_feature_group(
        name=feature_group_metadata.name,
        version=feature_group_metadata.version,
        description=feature_group_metadata.description,
        primary_key=feature_group_metadata.primary_key,
        event_time=feature_group_metadata.event_time,
        online_enabled=feature_group_metadata.online_enabled
    )

def get_or_create_feature_view(
    feature_view_metadata: FeatureViewConfig
) -> hsfs.feature_view.FeatureView:
    """Connects to the feature store and returns the feature view. Creates it if it does not exist."""
    
    feature_store = get_feature_store()

    try:
        feature_view = feature_store.get_feature_view(
            name=feature_view_metadata.name,
            version=feature_view_metadata.version,
        )
        logging.info(f"✅ Feature view `{feature_view_metadata.name}` (v{feature_view_metadata.version}) already exists.")
        return feature_view

    except Exception as e:
        logging.info(f"ℹ️ Feature view `{feature_view_metadata.name}` not found, trying to create it...")

        try:
            feature_group = feature_store.get_feature_group(
                name=feature_view_metadata.feature_group.name,
                version=feature_view_metadata.feature_group.version
            )

            feature_store.create_feature_view(
                name=feature_view_metadata.name,
                version=feature_view_metadata.version,
                query=feature_group.select_all()
            )

            logging.info(f"✅ Feature view `{feature_view_metadata.name}` created successfully.")
            return feature_store.get_feature_view(
                name=feature_view_metadata.name,
                version=feature_view_metadata.version
            )

        except Exception as creation_error:
            logging.error(f"❌ Failed to create or retrieve feature view `{feature_view_metadata.name}`: {creation_error}")
            raise