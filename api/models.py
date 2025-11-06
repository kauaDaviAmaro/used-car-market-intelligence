"""Pydantic models for API request/response."""
from pydantic import BaseModel, Field
from typing import Optional

class CarFeatures(BaseModel):
    """Car features for price prediction."""
    ano: float = Field(..., description="Year of the car")
    quilometragem: Optional[float] = Field(None, description="Mileage in km")
    motor: Optional[float] = Field(None, description="Engine size")
    marca: str = Field(..., description="Car brand")
    state: str = Field(..., description="State (e.g., SP, RJ)")
    cambio: Optional[str] = Field(None, description="Transmission type (Automático, Manual)")
    combustivel: Optional[str] = Field(None, description="Fuel type")
    direcao: Optional[str] = Field(None, description="Steering type")
    cor: Optional[str] = Field(None, description="Color")
    tipo_de_veiculo: Optional[str] = Field(None, description="Vehicle type")
    tipo_de_direcao: Optional[str] = Field(None, description="Steering type detail")
    possui_kit_gnv: Optional[str] = Field(None, description="Has GNV kit (Sim, Não)")
    portas: Optional[float] = Field(None, description="Number of doors")
    potencia: Optional[float] = Field(None, description="Power")
    final_de_placa: Optional[float] = Field(None, description="License plate final digit")
    
    # Boolean features - using a list to avoid repetition
    air_bag: Optional[bool] = Field(False, description="Has airbag")
    ar_condicionado: Optional[bool] = Field(False, description="Has air conditioning")
    alarme: Optional[bool] = Field(False, description="Has alarm")
    controle_automatico_de_velocidade: Optional[bool] = Field(False, description="Has cruise control")
    trava_eletrica: Optional[bool] = Field(False, description="Has electric locks")
    vidro_eletrico: Optional[bool] = Field(False, description="Has electric windows")
    ipva_pago: Optional[bool] = Field(False, description="IPVA paid")
    pneus_novos: Optional[bool] = Field(False, description="New tires")
    sensor_de_re: Optional[bool] = Field(False, description="Reverse sensor")
    historico_veicular: Optional[bool] = Field(False, description="Vehicle history")
    aceita_trocas: Optional[bool] = Field(False, description="Accepts trades")
    garantia_de_3_meses: Optional[bool] = Field(False, description="3 month warranty")
    laudo_veicular: Optional[bool] = Field(False, description="Vehicle report")
    camera_de_re: Optional[bool] = Field(False, description="Reverse camera")
    com_manual: Optional[bool] = Field(False, description="With manual")
    com_garantia: Optional[bool] = Field(False, description="With warranty")
    entrega_do_veiculo: Optional[bool] = Field(False, description="Vehicle delivery")
    computador_de_bordo: Optional[bool] = Field(False, description="On-board computer")
    transferencia_de_documentacao: Optional[bool] = Field(False, description="Documentation transfer")
    carro_de_leilao: Optional[bool] = Field(False, description="Auction car")
    rodas_de_liga_leve: Optional[bool] = Field(False, description="Alloy wheels")
    unico_dono: Optional[bool] = Field(False, description="Single owner")
    conexao_usb: Optional[bool] = Field(False, description="USB connection")
    bancos_de_couro: Optional[bool] = Field(False, description="Leather seats")
    interface_bluetooth: Optional[bool] = Field(False, description="Bluetooth interface")
    higienizacao_do_veiculo: Optional[bool] = Field(False, description="Vehicle sanitization")
    tracao_4x4: Optional[bool] = Field(False, description="4x4 traction")
    tanque_cheio: Optional[bool] = Field(False, description="Full tank")
    laudo_cautelar: Optional[bool] = Field(False, description="Cautionary report")
    chave_reserva: Optional[bool] = Field(False, description="Spare key")
    som: Optional[bool] = Field(False, description="Sound system")
    com_multas: Optional[bool] = Field(False, description="With fines")
    primeira_revisao_gratis: Optional[bool] = Field(False, description="First service free")
    blindado: Optional[bool] = Field(False, description="Armored")
    navegador_gps: Optional[bool] = Field(False, description="GPS navigator")
    revisoes_feitas_em_concessionaria: Optional[bool] = Field(False, description="Services done at dealership")
    ipva_gratis: Optional[bool] = Field(False, description="Free IPVA")
    apoio_na_documentacao: Optional[bool] = Field(False, description="Documentation support")
    teto_solar: Optional[bool] = Field(False, description="Sunroof")
    veiculo_em_financiamento: Optional[bool] = Field(False, description="Vehicle in financing")
    garantia_3_meses: Optional[bool] = Field(False, description="3 month warranty")
    veiculo_quitado: Optional[bool] = Field(False, description="Paid off vehicle")
    financiado: Optional[bool] = Field(False, description="Financed")
    garantia_do_motor: Optional[bool] = Field(False, description="Engine warranty")
    volante_multifuncional: Optional[bool] = Field(False, description="Multifunctional steering wheel")
    com_garantia_de_fabrica: Optional[bool] = Field(False, description="Factory warranty")
    com_chave_reserva: Optional[bool] = Field(False, description="With spare key")

    class Config:
        json_schema_extra = {
            "example": {
                "ano": 2020,
                "quilometragem": 50000,
                "motor": 1.6,
                "marca": "Chevrolet",
                "state": "SP",
                "cambio": "Automático",
                "combustivel": "Flex",
                "direcao": "Hidráulica",
                "cor": "Branco",
                "portas": 4,
                "bancos_de_couro": True,
                "ar_condicionado": True,
                "blindado": False
            }
        }

