"""Multi-language alert message catalog and localized voice announcements.

Provides standardized, firm, and understandable messages across alert stages.
Default language is English, with verified Hindi and Marathi localizations.

Safety principles:
- Short, clear, and firm without causing unnecessary panic or abrupt braking.
- Instructs the driver to stop safely when drowsy.
- Never claims the system guarantees accident prevention.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from backend.temporal_state import AlertStage


# Standard alert texts across stages and supported languages
ALERT_MESSAGES: Dict[str, Dict[AlertStage, str]] = {
    "en": {
        AlertStage.STAGE_0_NORMAL: "Normal monitoring active. Stay attentive.",
        AlertStage.STAGE_1_EARLY: "Please stay alert. Keep your eyes on the road.",
        AlertStage.STAGE_2_DROWSY: "Warning! You may be getting drowsy. Please find a safe place to stop.",
        AlertStage.STAGE_3_CRITICAL: "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.",
    },
    "hi": {
        AlertStage.STAGE_0_NORMAL: "निगरानी सक्रिय है। सतर्क रहें।",
        AlertStage.STAGE_1_EARLY: "कृपया सतर्क रहें। अपनी आँखें सड़क पर रखें।",
        AlertStage.STAGE_2_DROWSY: "चेतावनी! आपको नींद आ रही हो सकती है। कृपया रुकने के लिए सुरक्षित स्थान खोजें।",
        AlertStage.STAGE_3_CRITICAL: "खतरा! यथाशीघ्र सुरक्षित रूप से गाड़ी रोकें। झपकी की स्थिति में गाड़ी न चलाएं।",
    },
    "mr": {
        AlertStage.STAGE_0_NORMAL: "निरीक्षण सुरू आहे. लक्ष ठेवा.",
        AlertStage.STAGE_1_EARLY: "कृपया सावध रहा. आपले लक्ष रस्त्यावर ठेवा.",
        AlertStage.STAGE_2_DROWSY: "इशारा! तुम्हाला झोप येत असावी. कृपया थांबण्यासाठी सुरक्षित जागा शोधा.",
        AlertStage.STAGE_3_CRITICAL: "धोका! लवकरात लवकर सुरक्षितपणे गाडी बाजूला घ्या. झोपेत असताना गाडी चालवू नका.",
    },
}

ALERT_TITLES: Dict[str, Dict[AlertStage, str]] = {
    "en": {
        AlertStage.STAGE_0_NORMAL: "ATTENTIVE",
        AlertStage.STAGE_1_EARLY: "STAGE 1 - ATTENTION WARNING",
        AlertStage.STAGE_2_DROWSY: "STAGE 2 - DROWSINESS WARNING",
        AlertStage.STAGE_3_CRITICAL: "STAGE 3 - CRITICAL WARNING",
    },
    "hi": {
        AlertStage.STAGE_0_NORMAL: "सतर्क",
        AlertStage.STAGE_1_EARLY: "चरण १ - ध्यान दें",
        AlertStage.STAGE_2_DROWSY: "चरण २ - सुस्ती चेतावनी",
        AlertStage.STAGE_3_CRITICAL: "चरण ३ - गंभीर सुरक्षा चेतावनी",
    },
    "mr": {
        AlertStage.STAGE_0_NORMAL: "सावध",
        AlertStage.STAGE_1_EARLY: "टप्पा १ - लक्ष द्या",
        AlertStage.STAGE_2_DROWSY: "टप्पा २ - डुलकी इशारा",
        AlertStage.STAGE_3_CRITICAL: "टप्पा ३ - गंभीर सुरक्षा इशारा",
    },
}


class AlertMessageCatalog:
    """Catalog manager for multi-language alert text and voice prompts."""

    DEFAULT_LANGUAGE = "en"
    SUPPORTED_LANGUAGES: List[str] = ["en", "hi", "mr"]

    @classmethod
    def get_message(cls, stage: AlertStage, language: str = DEFAULT_LANGUAGE) -> str:
        """Retrieve localized spoken message for a given alert stage."""
        lang = language.lower() if language.lower() in ALERT_MESSAGES else cls.DEFAULT_LANGUAGE
        return ALERT_MESSAGES[lang].get(
            stage,
            ALERT_MESSAGES[cls.DEFAULT_LANGUAGE][stage],
        )

    @classmethod
    def get_title(cls, stage: AlertStage, language: str = DEFAULT_LANGUAGE) -> str:
        """Retrieve localized display title for a given alert stage."""
        lang = language.lower() if language.lower() in ALERT_TITLES else cls.DEFAULT_LANGUAGE
        return ALERT_TITLES[lang].get(
            stage,
            ALERT_TITLES[cls.DEFAULT_LANGUAGE][stage],
        )

    @classmethod
    def is_language_supported(cls, language: str) -> bool:
        """Check if requested language code is supported."""
        return language.lower() in cls.SUPPORTED_LANGUAGES
