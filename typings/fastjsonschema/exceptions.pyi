class JsonSchemaException(ValueError): ...

class JsonSchemaValueException(JsonSchemaException):
    message: str
    value: object
    name: str
    definition: object
    rule: str
    def __init__(
        self,
        message: str,
        value: object = ...,
        name: str = ...,
        definition: object = ...,
        rule: str = ...,
    ) -> None: ...
    @property
    def path(self) -> list[str]: ...
    @property
    def rule_definition(self) -> object: ...

class JsonSchemaDefinitionException(JsonSchemaException): ...
