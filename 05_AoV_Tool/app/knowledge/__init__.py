
from app.knowledge.base import KnowledgeBase

# Global Singleton
_kb_instance = None

def get_knowledge_base():
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
