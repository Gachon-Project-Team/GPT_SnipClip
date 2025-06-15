import styled from 'styled-components';

export const InputContainer = styled.div`
    width: 80%;
    min-height: 300px;
    background-color: white;
    color: black;
    display: flex;
    flex-direction: column;
    border-radius: 15px;
    padding: 20px;
`
export const KeywordInputForm = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding: 20px;
  background-color: whitesmoke;
  border-radius: 10px;
  min-height: 300px;
  transition: all 0.3s ease;
  height: ${(props) => (props.isProcessing ? '500px' : 'auto')};

  /* border : solid yellow 3px; */
`;

export const TextElement = styled.h1`
  color: black;
  font-size: 24px;
  margin-bottom: 50px;
`;

export const FormContainer = styled.div`
  display: flex;
  flex-direction: row;
  gap: 5px;
  width: 100%;
`;

export const InputField = styled.input`
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 5px;
  width: 100%;
`;

export const ProgressBarContainer = styled.div`
  width: 100%;
  background-color: #f3f3f3;
  border-radius: 5px;
  margin-top: 20px;
`;

export const DownloadButton = styled.button`
  margin-top: 20px;
  padding: 4px 10px;
  font-size: 16px;
  font-weight: bold;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  &:hover {
    background-color: #218838;
  }
  &:active {
    background-color: #1e7e34;
  }
`;